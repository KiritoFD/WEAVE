import pathlib

p = pathlib.Path("sbflow/train5.py")
s = p.read_text(encoding="utf-8")
start = s.index("def _apply_config(p, argv):")
end = s.index("# --------------------------------------------------------------------------- #\ndef main():", start)
new = '''def _apply_config(p, argv):
    """Load a flat JSON config and feed it to argparse as explicit CLI overrides.

    Keys are argparse *dest* names (underscores). Unknown / bookkeeping keys are
    ignored rather than rejected, so the suite driver can keep its metadata in the
    same file. Explicit command-line arguments take precedence (appended last).
    """
    import json as _json

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, rest = pre.parse_known_args(argv)
    if not known.config:
        return p.parse_args(argv)
    cfg = _json.loads(Path(known.config).read_text(encoding="utf-8"))

    dests = {a.dest: a for a in p._actions if a.dest != "help"}
    bool_true = {d for d, a in dests.items() if isinstance(a, argparse._StoreTrueAction)}
    skip = {"id", "group", "note", "fork_from", "priority", "quick200", "full750",
            "config"}
    # store_false flags share a dest with a store_true option; map a false value back
    # to the negative CLI flag. (single_pair / frequency_stats in JSON are redundant
    # aliases and are simply ignored because they are not real dests.)
    neg_alias = {"mixed_batch": "--single-pair",
                 "equal_weight_stats": "--frequency-stats"}
    pos_alias = {"mixed_batch": "--mixed-batch",
                 "equal_weight_stats": "--equal-weight-stats"}
    extra = []
    for k, v in cfg.items():
        k = str(k).replace("-", "_")
        if k in skip or v is None or k not in dests:
            continue
        key = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if k in neg_alias:
                extra.append(neg_alias[k] if not v else pos_alias[k])
            elif k in bool_true and v:
                extra.append(key)
            # store_true dest with false value: keep the parser default
        elif isinstance(v, list):
            extra.extend([key] + [str(x) for x in v])
        else:
            extra.extend([key, str(v)])
    return p.parse_args(extra + rest)


'''
p.write_text(s[:start] + new + s[end:], encoding="utf-8")
print("rewrote _apply_config")
