import os

def print_stats(base):
    for a in sorted(["coco", "custom"]):
        for b in sorted(os.listdir(os.path.join(base, a))):
            dir_ab = os.path.join(base, a, b)
            entries = sorted(os.listdir(dir_ab))

            describe = sum("describe" in c for c in entries)
            detect = sum("detect" in c for c in entries)
            match = sum("match" in c for c in entries)
            eval_count = sum("match" in c and "evaluation.json" in os.listdir(os.path.join(dir_ab, c)) for c in entries)

            if describe > 0 and describe == detect and describe == match and describe == eval_count:
                print(f"{dir_ab}: All '{describe}'.")
            else:
                print(f"{dir_ab}:\n\tdescribe: {describe}\n\tdetect: {detect}\n\tmatch: {match}\n\teval: {eval_count}")

def get_paths(base, mode, verbose=False):
    assert mode in ["detect", "match", "eval"]
    result = []
    for a in sorted(["coco", "custom"]):
        for b in sorted(os.listdir(os.path.join(base, a))):
            dir_ab = os.path.join(base, a, b)
            entries = sorted(os.listdir(dir_ab))
            has_describe = any("describe" in c for c in entries)

            if mode == "detect" and not has_describe:
                if verbose:
                    print('misses describe: "' + dir_ab + '/",')
                continue

            has_detect = any("detect" in c for c in entries)

            if mode == "detect" and has_detect:
                continue

            if mode == "match" and not has_detect:
                if verbose:
                    print('misses detect: "' + dir_ab + '/",')
                continue

            has_match = any("match" in c for c in entries)

            if mode == "match" and has_match:
                continue

            if mode == "eval" and not has_match:
                if verbose:
                    print('misses match: "' + dir_ab + '/",')
                continue

            has_eval = any("match" in c and "evaluation.json" in os.listdir(os.path.join(dir_ab, c)) for c in entries)

            if mode == "eval" and has_eval:
                continue

            for c in entries:
                path = os.path.join(dir_ab, c)
                if mode == "detect" and path.endswith("describe"):
                    print(path)
                    result.append(path)
                elif mode == "match" and path.endswith("detect"):
                    print(path)
                    result.append(path)
                elif mode == "eval" and path.endswith("match"):
                    print(path)
                    result.append(path)
    return result
