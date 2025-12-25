def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip("\n")
            if s.strip() == "":
                continue
            lines.append(s)
    return lines
