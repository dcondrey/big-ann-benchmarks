import os

if os.environ.get("HMS_STREAM_PROBE") == "1":
    from neurips23.streaming.hms.streaming_probe import register_dataset

    register_dataset()
