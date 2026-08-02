import cv2
import glob
import os
import numpy as np

STEP = 50
RANK_W, RANK_H = 45, 45
# The suit pip sits beside the rank digit in the same top corner index, not
# below it: mirror the rank patch's box onto the right half of the card.
# (The old offset (5, 42, 35, 22) undershot into rank-glyph tail pixels
# instead of the actual suit icon - see suit-detection investigation notes.)
SUIT_X_OFF, SUIT_Y_OFF = 45, 0
SUIT_W, SUIT_H = 45, 45
PAD = 15
HIDDEN_CARD_H = 30
TOP_RESIDUAL_TOLERANCE = 5

# fixed x-start positions for each tableau column (UI layout doesn't move)
TABLEAU_X = [10, 111, 212, 313, 414, 512, 611]
TABLEAU_Y_TOP = 507
COL_WIDTH = 95

# Confirmed live against the real device: the 4 boxes at this position are
# the foundation piles (dragging an out-of-sequence card there is rejected,
# and an exposed Ace has been observed auto-completing into one of these
# boxes) - not free cells. The game has no free cells at all.
FOUNDATION_X = [10, 110, 210, 310]
SLOT_Y = 303
SLOT_W, SLOT_H = 95, 90

# The waste pile (draw-3 stock reveal) renders fanned/overlapping in this
# UI - only the frontmost card shows in full, cards behind it show just a
# sliver of their own corner rank digit - so a single fixed x either misses
# cards or bleeds across several (a single x=472 crop was observed pulling
# in two different cards' digits at once). WASTE_SCAN_X0 starts just past
# foundation slot 4's right edge (310 + SLOT_W = 405) so foundation content
# already handled by read_slot() below can't bleed in and get mistaken for
# a waste card.
WASTE_SCAN_X0, WASTE_SCAN_X1 = 410, 650
WASTE_PEAK_THRESHOLD = 0.65

# A card's landing/score-popup animation can briefly obscure a foundation
# slot's rank+suit crop, producing a match against noise instead of the
# real card. This score cutoff is only a FIRST-line filter for the deepest
# garbage (confirmed live at 0.24-0.25 for fully covered slots): measured
# across the full Gameplay corpus, animation garbage also scores 0.32-0.72
# (frame_0006 read a K-of-spades at 0.32 over a slot that really held a 3)
# while genuine weak reads sit at 0.32-0.34, so score alone CANNOT separate
# the two. The real defense is the temporal validation in
# solitaire_auto_bot.py (foundations only grow, suits never change), which
# rejects impossible jumps regardless of score.
MIN_FOUNDATION_SCORE = 0.3

# Confirmed live by tapping through an entire stock cycle: 24 cards, drawn
# 3 at a time (8 taps to exhaust), then an unlimited, deterministic redeal
# (the same 24 cards return in the same order every cycle). The deal shape
# is always the classic 1-7 tableau triangle (28 cards), so stock is always
# the fixed remainder: 52 - 28 = 24.
STOCK_TOTAL = 24
# Confirmed live: tapping this position both draws the next 3 cards and,
# once the stock is exhausted, triggers the redeal - same physical button,
# the game itself decides which behavior applies.
STOCK_TAP_X, STOCK_TAP_Y = 662, 368

from pathlib import Path

def load_templates(folder):
    # Resolve folder relative to this module file so imports from other
    # working directories still find the packaged template images.
    base = Path(__file__).parent
    search_dir = base / folder
    t = {}
    if not search_dir.exists():
        # graceful: return empty dict and let callers handle missing templates
        return t
    for path in glob.glob(str(search_dir / "*.png")):
        name = os.path.splitext(os.path.basename(path))[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # skip unreadable files
            continue
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t[name] = binary
    return t

TEMPLATES = load_templates("templates")
TEMPLATES_LAST = load_templates("templates_last")
SUIT_TEMPLATES = load_templates("suit_templates")

# Red-ink gate: red hue (OpenCV hue wraps at 180 - this game's red ink is
# crimson at h~168, so the upper band must reach down to 150) AND heavily
# saturated. Black-ink gate: dark and near-neutral (black ink and its gray
# anti-aliasing have low saturation). Dark background content - green felt
# (h~60-90, saturated), brown frame shadows, gold/amber popup glow - fails
# both gates and is ignored entirely.
RED_HUE_LO, RED_HUE_HI = 15, 150
RED_SAT_MIN = 150
BLACK_SAT_MAX = 60
# Verdict cutoff: red-pixel share of ALL dark pixels. Corpus-measured
# extremes are 0.276 (highest black pip, a straddled crop polluted by a
# neighboring red pip) vs 0.308 (lowest genuine red pip, same straddle
# problem in reverse) - NOT comfortably separated, which is why the
# ambiguity band below exists.
RED_SHARE_CUTOFF = 0.3
# Confidence: share of red ink among ink pixels only (background excluded,
# so occlusion can't dilute it). Measured across all 2,968 corpus crops,
# 99% of reads sit outside [0.20, 0.55]; every crop inside the band was
# either a crop straddling two cards' pips or animation garbage - exactly
# the reads that must not be trusted silently.
AMBIG_INK_FRAC_LO, AMBIG_INK_FRAC_HI = 0.20, 0.55
MIN_INK_PIXELS = 5

def classify_suit_color(patch):
    """Classify a suit-pip crop as RED or BLACK.

    Returns (color, confident). color is "RED"/"BLACK"/"?"; confident is
    False when the ink evidence is mixed or absent - a crop straddling two
    cards, an animation graphic covering the pip, or an empty crop - and
    the caller must treat the read as unresolved (zero its score) rather
    than trust the guess. A fully-covered slot (e.g. a "+20" popup) has no
    ink at all, so this also catches obscured reads whose template score
    happens to look plausible.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark = gray < 200
    n_dark = dark.sum()
    if n_dark < 5:
        return "?", False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    red_ink = dark & ((h <= RED_HUE_LO) | (h >= RED_HUE_HI)) & (s > RED_SAT_MIN)
    black_ink = dark & (s < BLACK_SAT_MAX)
    n_red, n_black = int(red_ink.sum()), int(black_ink.sum())
    if n_red + n_black < MIN_INK_PIXELS:
        return "?", False
    color = "RED" if n_red / n_dark > RED_SHARE_CUTOFF else "BLACK"
    ink_frac = n_red / (n_red + n_black)
    confident = (color == "RED" and ink_frac >= AMBIG_INK_FRAC_HI) or \
                (color == "BLACK" and ink_frac <= AMBIG_INK_FRAC_LO)
    return color, confident

def match_rank(patch, template_set):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(binary, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)

    best_name, best_score = "?", -1
    for name, tmpl in template_set.items():
        if tmpl.shape[0] > padded.shape[0] or tmpl.shape[1] > padded.shape[1]:
            continue
        result = cv2.matchTemplate(padded, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_name = name
    # Rank templates support multiple named variants per rank, using the
    # same file scheme suits already use ("2_live.png" is a variant of
    # "2"): collapse the winning variant back to its base name.
    # match_suit's own collapse then becomes a no-op, which is fine.
    return best_name.split("_")[0], best_score

# Suit templates (S/H/D/C) are matched with the exact same shape-matching
# approach as rank templates - match_rank() is generic over its template
# set, so it's reused as-is rather than duplicating the logic. Spade and
# club in particular are close enough in shape that a single example per
# suit produces near-tied scores (measured ~0.92 vs ~0.91 on a real
# misclassification), so suit templates support multiple named variants
# per suit (files "S_1.png", "S_2.png", ...) and match_suit() collapses
# the winning variant back down to its base suit letter.
#
# color is required and narrows the template set to the 2 same-color
# candidates (S/C for BLACK, H/D for RED) before matching. classify_suit_color
# is reliable on its own (color, unlike shape, isn't confusable), so this
# guarantees a black card can never be scored against a red template (or
# vice versa) - a cross-color mismatch measured in practice (a black
# card's best raw template score sometimes lands on a red suit) that pure
# shape-matching over all 4 templates can't rule out on its own.
def match_suit(patch, color, template_set=SUIT_TEMPLATES):
    if color not in ("BLACK", "RED"):
        # Fail closed: with no color evidence, matching across all 4
        # templates is exactly the cross-color mismatch the comment above
        # says color-gating exists to prevent (near-tied spade/heart-class
        # shapes), and the result could still carry a passable score.
        return "?", 0.0
    candidates = {"S", "C"} if color == "BLACK" else {"H", "D"}
    narrowed = {
        name: tmpl for name, tmpl in template_set.items() if name.split("_")[0] in candidates
    }
    name, score = match_rank(patch, narrowed)
    return name.split("_")[0], score

def detect_waste_slots(img):
    """Dynamically locate each visible waste-pile card's corner rank digit,
    instead of reading a single fixed x-coordinate. Cards behind the
    frontmost one only reveal a sliver of their own corner index, so
    contour/bbox segmentation can't separate them (overlapping cards fuse
    into one blob) - this scans the whole waste band with the existing
    rank templates via sliding-window template matching (cv2.matchTemplate
    naturally scores every x-shift in one call) and returns one (x, rank,
    score) per local peak above threshold. An empty waste (nothing drawn
    yet, or just after a redeal) simply produces no peak, so this adapts
    to however many of the up-to-3 drawn cards are currently showing
    without needing to know that count in advance.
    """
    def scan(template_set, h):
        strip = img[SLOT_Y:SLOT_Y+h, WASTE_SCAN_X0:WASTE_SCAN_X1]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        padded = cv2.copyMakeBorder(binary, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
        best_score, best_name = None, None
        for name, tmpl in template_set.items():
            if tmpl.shape[0] > padded.shape[0] or tmpl.shape[1] > padded.shape[1]:
                continue
            result = cv2.matchTemplate(padded, tmpl, cv2.TM_CCOEFF_NORMED)
            row = result.max(axis=0)
            if best_score is None:
                best_score, best_name = row, np.full(row.shape, name, dtype=object)
            else:
                mask = row > best_score
                best_score = np.where(mask, row, best_score)
                best_name[mask] = name
        return best_score, best_name

    # TEMPLATES catches piles partially covered by the fan (compact corner
    # index only); TEMPLATES_LAST catches the frontmost, fully-open pile
    # (bigger center digit) - take whichever scores higher at each x.
    score_a, name_a = scan(TEMPLATES, RANK_H)
    score_b, name_b = scan(TEMPLATES_LAST, 90)
    length = min(len(score_a), len(score_b))
    score = np.maximum(score_a[:length], score_b[:length])
    name = np.where(score_a[:length] >= score_b[:length], name_a[:length], name_b[:length])

    slots = []
    i = 0
    while i < length:
        if score[i] > WASTE_PEAK_THRESHOLD:
            j = i
            while j < length and score[j] > WASTE_PEAK_THRESHOLD:
                j += 1
            peak = i + int(np.argmax(score[i:j]))
            x = WASTE_SCAN_X0 + peak - PAD
            # A peak inside the left padding margin (x < WASTE_SCAN_X0)
            # is matching zero-padding plus band-edge bleed, not a card -
            # observed live as a phantom score-0 read at x=401 that taxed
            # every cycle with unreliable-frame retries.
            if peak >= PAD:
                slots.append((x, str(name[peak]).split("_")[0],
                              round(float(score[peak]), 2)))
            i = j
        else:
            i += 1
    return slots

def detect_column_height(img, x):
    """Detect how tall the card stack in this column currently is, using
    white-region contour detection (same approach as find_cards.py).

    The white mask only picks up face-up (revealed) cards; face-down cards at
    the top of a column render as a uniform ~30px sliver per card that the
    mask doesn't see, so the revealed region's top edge sits that much lower
    than the column origin. Returns (height, hidden_count, reliable):
    - height: pixel bottom of the revealed region (unchanged meaning from
      before hidden-card support)
    - hidden_count: how many face-down cards sit above the revealed region,
      inferred from that top offset
    - reliable: False when the top offset doesn't cleanly fit a whole number
      of hidden cards (~30px each) - a sign of a mid-animation render rather
      than a real, stable hidden-card count
    """
    col_slice = img[TABLEAU_Y_TOP:, x:x+COL_WIDTH]
    hsv = cv2.cvtColor(col_slice, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, True

    max_y, min_y = 0, None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 30 and ch > 20:
            max_y = max(max_y, cy + ch)
            min_y = cy if min_y is None else min(min_y, cy)

    if min_y is None:
        return 0, 0, True

    hidden_count = round(min_y / HIDDEN_CARD_H)
    residual = abs(min_y - hidden_count * HIDDEN_CARD_H)
    reliable = residual <= TOP_RESIDUAL_TOLERANCE
    return max_y, hidden_count, reliable  # bottom pixel of the revealed region

def read_board(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(frame_path)

    board = {}
    for col_idx, x in enumerate(TABLEAU_X):
        height, hidden_count, reliable = detect_column_height(img, x)
        col_cards = []

        if height < 100:  # empty or noise, treat as empty column
            board[f"col{col_idx}"] = col_cards
            continue

        revealed_span = height - hidden_count * HIDDEN_CARD_H
        num_rows = round((revealed_span - 135) / 50) + 1
        num_rows = max(1, num_rows)

        # face-down cards have no readable rank; represent them as unknown
        # rather than feeding their pixels into the rank matcher
        col_cards.extend({"rank": "?", "suit": "?", "color": "?", "score": 0.0} for _ in range(hidden_count))

        y_start = TABLEAU_Y_TOP + hidden_count * HIDDEN_CARD_H
        for row in range(num_rows):
            y = y_start + row * STEP
            is_last = (row == num_rows - 1)

            # The suit pip lives in the same top-corner index box regardless
            # of whether this is a compact stacked row or a fully-revealed
            # "last" card - both render the small rank+suit index at the
            # same relative position, so one crop formula covers both.
            suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
            color, color_confident = classify_suit_color(suit_patch)
            suit, suit_score = match_suit(suit_patch, color)

            if is_last:
                rank_patch = img[y:y+90, x:x+95]
                name, score = match_rank(rank_patch, TEMPLATES_LAST)
            else:
                rank_patch = img[y:y+RANK_H, x:x+RANK_W]
                name, score = match_rank(rank_patch, TEMPLATES)

            score = min(score, suit_score)

            # a top offset that doesn't cleanly fit a whole number of hidden
            # cards means this frame is mid-animation, not a stable state -
            # every row crop here is misaligned regardless of match_rank's score
            # mixed red/black ink evidence (a crop straddling two cards, or
            # an animation graphic over the pip) means the color - and so
            # the suit - is a coin flip; zero the score so state builders
            # treat the card as unresolved instead of trusting a guess
            if not reliable or not color_confident:
                score = 0.0

            col_cards.append({"rank": name, "suit": suit, "color": color, "score": round(float(score), 2)})

        board[f"col{col_idx}"] = col_cards

    def read_slot(x):
        patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if gray.std() < 15:
            return None
        name, score = match_rank(patch, TEMPLATES_LAST)
        suit_patch = img[SLOT_Y+SUIT_Y_OFF:SLOT_Y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
        color, color_confident = classify_suit_color(suit_patch)
        suit, suit_score = match_suit(suit_patch, color)
        score = min(score, suit_score)
        # reliable is the producer-side trust verdict every consumer should
        # honor (build_solver_state does): an occupied-looking slot whose
        # ink evidence is mixed/absent or whose score is below the
        # first-line cutoff must not be trusted as a real card this frame.
        reliable = bool(color_confident and score >= MIN_FOUNDATION_SCORE)
        return {"rank": name, "suit": suit, "color": color,
                "score": round(float(score), 2), "reliable": reliable}

    # Foundation piles are 4 separate, non-overlapping boxes (unlike the
    # fanned waste pile below), so the same per-x read_slot() used for
    # tableau "last card" reads applies directly - one clean crop per box.
    board["foundation"] = [read_slot(x) for x in FOUNDATION_X]

    waste = []
    for x, rank, rank_score in detect_waste_slots(img):
        suit_patch = img[SLOT_Y+SUIT_Y_OFF:SLOT_Y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
        color, color_confident = classify_suit_color(suit_patch)
        suit, suit_score = match_suit(suit_patch, color)
        score = min(rank_score, suit_score)
        if not color_confident:
            score = 0.0
        # keep the source x so callers can find the current playable
        # (frontmost = last = highest x) card's tap/swipe position
        waste.append({"rank": rank, "suit": suit, "color": color, "score": round(float(score), 2), "x": x})
    board["waste"] = waste

    return board
