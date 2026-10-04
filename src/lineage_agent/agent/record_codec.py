"""JSON originals with a fixed, non-executable encoding for safe opaque values."""

import base64
import json
import math
from typing import Any


def _json(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        # Preserve invalid protocol strings for admission to reject later. The
        # journal itself is still valid UTF-8 JSON, including lone surrogates.
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _pack(value: Any, stack: set[int]) -> list[Any]:
    kind = type(value)
    if value is None:
        return ["null", None]
    if kind in (str, bool, int):
        return [{str: "str", bool: "bool", int: "int"}[kind], value]
    if kind is float:
        return ["float", value] if math.isfinite(value) else ["nonfinite", "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")]
    if kind in (bytes, bytearray):
        return ["bytes" if kind is bytes else "bytearray", base64.b64encode(value).decode("ascii")]
    if kind not in (dict, list, tuple, set, frozenset) or id(value) in stack:
        raise ValueError("unsupported original value")
    stack.add(id(value))
    try:
        if kind is dict:
            return ["dict", [[_pack(key, stack), _pack(item, stack)] for key, item in value.items()]]
        items = [_pack(item, stack) for item in value]
        if kind in (set, frozenset):
            items.sort(key=_json)
        return [{list: "list", tuple: "tuple", set: "set", frozenset: "frozenset"}[kind], items]
    finally:
        stack.remove(id(value))


def _unpack(node: Any) -> Any:
    if type(node) is not list or len(node) != 2 or type(node[0]) is not str:
        raise ValueError("invalid opaque value")
    tag, value = node
    expected = {"str": str, "bool": bool, "int": int, "float": float}
    if tag == "null" and value is None:
        return None
    if tag in expected and type(value) is expected[tag] and (tag != "float" or math.isfinite(value)):
        return value
    if tag == "nonfinite" and type(value) is str and value in {"nan", "inf", "-inf"}:
        return float(value)
    if tag in {"bytes", "bytearray"} and type(value) is str:
        result = base64.b64decode(value, validate=True)
        return result if tag == "bytes" else bytearray(result)
    if tag in {"list", "tuple", "set", "frozenset"} and type(value) is list:
        values = [_unpack(item) for item in value]
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag](values)
    if tag == "dict" and type(value) is list:
        result = {}
        for pair in value:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("invalid mapping pair")
            key, item = _unpack(pair[0]), _unpack(pair[1])
            if key in result:
                raise ValueError("duplicate mapping key")
            result[key] = item
        return result
    raise ValueError("unsupported opaque tag")


def encode_record(record: dict[str, Any]) -> bytes:
    opaque: list[dict[str, Any]] = []
    stack: set[int] = set()

    def clean(value: Any, path: list[Any]) -> Any:
        kind = type(value)
        if value is None or kind in (str, bool, int) or (kind is float and math.isfinite(value)):
            return value
        if kind not in (dict, list) or (kind is dict and any(type(key) is not str for key in value)):
            opaque.append({"path": path, "encoding": _pack(value, stack)})
            return None
        if id(value) in stack:
            raise ValueError("cyclic original value")
        stack.add(id(value))
        try:
            if kind is dict:
                return {key: clean(value[key], path + [key]) for key in sorted(value)}
            return [clean(item, path + [index]) for index, item in enumerate(value)]
        finally:
            stack.remove(id(value))

    if type(record) is not dict:
        raise ValueError("original record must be an object")
    body = {"format_version": 1, "record": clean(record, []), "opaque_values": opaque}
    return _json(body) + b"\n"


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("invalid JSON constant")


def decode_record(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if (type(value) is not dict or set(value) != {"format_version", "record", "opaque_values"}
            or type(value["format_version"]) is not int or value["format_version"] != 1
            or type(value["record"]) is not dict or type(value["opaque_values"]) is not list):
        raise ValueError("invalid record envelope")
    result = value["record"]
    seen: set[tuple[Any, ...]] = set()
    for entry in value["opaque_values"]:
        if type(entry) is not dict or set(entry) != {"path", "encoding"} or type(entry["path"]) is not list or not entry["path"]:
            raise ValueError("invalid opaque path")
        path = entry["path"]
        if any(type(part) not in (str, int) for part in path):
            raise ValueError("invalid opaque path component")
        identity = tuple(path)
        if identity in seen:
            raise ValueError("duplicate opaque path")
        seen.add(identity)
        target = result
        for part in path[:-1]:
            if not ((type(target) is dict and type(part) is str and part in target)
                    or (type(target) is list and type(part) is int and 0 <= part < len(target))):
                raise ValueError("opaque path does not exist")
            target = target[part]
        last = path[-1]
        if not ((type(target) is dict and type(last) is str and last in target)
                or (type(target) is list and type(last) is int and 0 <= last < len(target))) or target[last] is not None:
            raise ValueError("opaque placeholder does not exist")
        target[last] = _unpack(entry["encoding"])
    # Enforce the exact encoder grammar, not just a decodable tagged object.
    if encode_record(result) != data:
        raise ValueError("noncanonical original record")
    return result
