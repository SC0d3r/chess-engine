import sys

def write_temp_line(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def clear_temp_line():
    # "\r" returns to the beginning of the line, "\033[K" clears to the end.
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()