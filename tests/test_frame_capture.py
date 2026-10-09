from PIL import Image, ImageDraw
from frame_capture import capture_settled_frame
from frame_capture import BackgroundCapture, FrameBuffer, decode_stream_header, VideoScreenrecordCapture
from unittest.mock import Mock
import struct
import pytest
import time
import threading


def frame(card_x=20, timer=False):
    image = Image.new("RGB", (720, 1600), (20, 110, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((card_x, 520, card_x + 90, 650), fill="white")
    if timer:
        draw.rectangle((100, 100, 200, 200), fill="white")
    return image


def capture_sequence(*frames):
    iterator = iter(frames)
    return lambda: next(iterator)


def test_reveal_animation_waits_for_settled_pair():
    final = frame(120)
    accepted, count, motion = capture_settled_frame(
        capture_sequence(frame(20), frame(70), final, final))
    assert accepted is final
    assert count == 4
    assert motion == 0


def test_clock_changes_do_not_delay_board_read():
    accepted, count, _ = capture_settled_frame(capture_sequence(frame(), frame(timer=True)))
    assert accepted is not None
    assert count == 2


def test_continuous_motion_is_never_sent_to_reader():
    accepted, count, motion = capture_settled_frame(
        capture_sequence(frame(20), frame(70), frame(120), frame(170)))
    assert accepted is None
    assert count == 4
    assert motion > 0


def test_failed_capture_breaks_consecutive_frame_pair():
    accepted, count, _ = capture_settled_frame(
        capture_sequence(frame(), None, frame(), frame()))
    assert accepted is not None
    assert count == 4


def test_unsupported_layout_is_left_to_existing_screen_guard():
    wrong = Image.new("RGB", (1080, 2400))
    accepted, count, _ = capture_settled_frame(capture_sequence(wrong))
    assert accepted is wrong
    assert count == 1


def test_cached_frame_before_last_gesture_is_not_reused():
    buffer = FrameBuffer()
    now = time.monotonic()
    buffer.publish(frame(), now - .3, now - .2)
    buffer.publish(frame(), now - .2, now - .1)
    assert buffer.wait(not_before=now - .05, timeout=0) is None
    buffer.publish(frame(), now, time.monotonic())
    assert buffer.wait(not_before=now, timeout=0).sequence == 3


def test_moving_latest_frame_invalidates_old_stable_frame():
    buffer = FrameBuffer()
    for image in (frame(), frame(), frame(120)):
        now = time.monotonic()
        buffer.publish(image, now, now)
    assert buffer.wait(timeout=0) is None
    now = time.monotonic()
    buffer.publish(frame(120), now, now)
    assert buffer.wait(timeout=0).sequence == 4


def test_disconnect_requires_new_consecutive_frames():
    buffer = FrameBuffer()
    for image in (frame(), frame(), None, frame()):
        now = time.monotonic()
        buffer.publish(image, now, now)
    assert buffer.wait(timeout=0) is None
    now = time.monotonic()
    buffer.publish(frame(), now, now)
    assert buffer.wait(timeout=0).sequence == 5


def test_a_capture_cannot_be_consumed_twice_or_after_expiring():
    buffer = FrameBuffer()
    for _ in range(2):
        now = time.monotonic()
        buffer.publish(frame(), now, now)
    assert buffer.wait(after_sequence=2, timeout=0) is None
    assert buffer.wait(max_age=-1, timeout=0) is None


def test_background_capture_runs_off_main_thread_and_stops():
    capture_threads = []
    image = frame()
    def capture():
        capture_threads.append(threading.get_ident())
        return image
    worker = BackgroundCapture(capture).start()
    try:
        assert worker.buffer.wait(timeout=2) is not None
        assert all(t != threading.get_ident() for t in capture_threads)
    finally:
        worker.stop()
    assert not worker.thread.is_alive()


def test_native_stream_header_preserves_resolution_and_rgb_length():
    header = struct.pack('<5I', 16 + 720*1600*3, 720, 1600, 720*3, 3)
    assert decode_stream_header(header) == (720, 1600, 720*1600*3)


@pytest.mark.parametrize('header', [
    (16, 720, 1600, 2160, 3), (16, 0, 1600, 0, 3),
    (16 + 720*1600*3, 720, 1600, 2160, 1),
    (16 + 720*1600*3, 720, 1600, 2164, 3),
])
def test_bad_stream_headers_cannot_allocate_or_desynchronize_frames(header):
    with pytest.raises(ValueError):
        decode_stream_header(struct.pack('<5I', *header))


def test_video_disconnect_closes_both_processes_before_reconnecting():
    source = VideoScreenrecordCapture(['adb'])
    source.process, source.decoder = Mock(), Mock()
    processes = [source.decoder, source.process]
    source._read_exact = Mock(side_effect=EOFError('disconnected'))
    source._terminate = Mock()
    with pytest.raises(EOFError):
        source()
    assert source.process is None and source.decoder is None
    assert [call.args[0] for call in source._terminate.call_args_list] == processes


def test_video_shutdown_stops_decoder_and_prevents_restart():
    source = VideoScreenrecordCapture(['adb'])
    source.process, source.decoder = Mock(), Mock()
    source._terminate = Mock()
    source.close()
    assert source._terminate.call_count == 2
    assert source() is None
