def format_time(t):

    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def generate_srt(segments):

    srt = ""

    for i, seg in enumerate(segments, start=1):

        start = format_time(seg["start"])
        end = format_time(seg["end"])

        srt += f"{i}\n"
        srt += f"{start} --> {end}\n"
        srt += f"{seg['text']}\n\n"

    return srt