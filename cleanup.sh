#!/bin/bash
set -e
cd ~/solittaire-bot

mkdir -p archive/old_versions
mkdir -p archive/debug_images
mkdir -p pipeline

mv -f board_reader.py board_reader2.py board_reader3.py board_reader4.py board_reader5.py \
   board_reader6.py board_reader7.py board_reader8.py board_reader9.py \
   board_reader_template.py board_reader_final.py \
   archive/old_versions/ 2>/dev/null || true

mv -f card_reader.py crop_test.py grid_debug.py finish_templates.py \
   extract_templates.py extract_templates_last.py \
   archive/old_versions/ 2>/dev/null || true

mv -f debug_corners archive/debug_images/ 2>/dev/null || true
mv -f card_boxes_debug.png grid_overlay.png archive/debug_images/ 2>/dev/null || true
mv -f col_0.png col_1.png col_2.png col_3.png col_4.png col_5.png col_6.png archive/debug_images/ 2>/dev/null || true

cp -f state_extractor.py pipeline/
cp -f find_cards.py pipeline/
cp -f board_reader_lib.py pipeline/
cp -f read_board.py pipeline/
cp -f suggest_move.py pipeline/

rm -f 500000

echo "DONE"
