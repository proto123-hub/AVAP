"""Source-level rules that keep the predecessor project's pathologies out.

- io discipline: cv2.imread/imwrite only inside avap/io_utils.py (L6/io).
- constants discipline: the shared change thresholds exist once (L3).
"""
import ast
import re
from pathlib import Path

AVAP_DIR = Path(__file__).resolve().parents[1] / "avap"


def _sources(exclude: set[str] = frozenset()) -> list[Path]:
    return [p for p in AVAP_DIR.rglob("*.py") if p.name not in exclude]


def test_no_direct_cv2_image_io():
    offenders = []
    for src in _sources(exclude={"io_utils.py"}):
        text = src.read_text(encoding="utf-8")
        if re.search(r"cv2\.(imread|imwrite)\b", text):
            offenders.append(src.name)
    assert not offenders, (
        f"cv2.imread/imwrite 직접 호출 금지 (한글 경로에서 조용히 실패): {offenders} — "
        "avap.io_utils.imread_u/imwrite_u를 쓸 것"
    )


def test_single_hsv_mask_generator():
    definitions = []
    inrange_modules = []
    for src in _sources():
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "make_mask":
                definitions.append(src.name)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "inRange"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "cv2"):
                inrange_modules.append(src.name)
    assert definitions == ["detection.py"], f"make_mask 단일 생성기 위반 (L6): {definitions}"
    assert set(inrange_modules) == {"detection.py"}, (
        f"HSV inRange는 make_mask 모듈에만 있어야 함 (L6): {inrange_modules}"
    )


def test_change_thresholds_defined_once():
    # 0.15 / 0.30 급변 임계가 constants.py 밖에 리터럴로 재정의되면 이중 정의 사고의 재판이다.
    offenders = []
    pattern = re.compile(r"(NOTICE_CHANGE_FRAC|LARGE_CHANGE_FRAC)\s*=")
    for src in _sources(exclude={"constants.py"}):
        if pattern.search(src.read_text(encoding="utf-8")):
            offenders.append(src.name)
    assert not offenders, f"급변 임계 재정의 금지 (L3): {offenders}"


def test_hsv_channel_scales_defined_once():
    offenders = []
    expected = [179.0, 255.0, 255.0]
    for src in _sources(exclude={"constants.py"}):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            values = [
                item.value for item in node.elts
                if isinstance(item, ast.Constant)
                and isinstance(item.value, (int, float))
                and not isinstance(item.value, bool)
            ]
            if len(values) == len(node.elts) and values == expected:
                offenders.append(f"{src.name}:{node.lineno}")
    assert not offenders, f"HSV 채널 스케일 재정의 금지 (L3): {offenders}"


def test_requirements_are_ascii_only():
    # 깨끗한 Windows venv의 번들 pip는 requirements 파일을 로케일 코덱(CP949)으로
    # 읽을 수 있어 비ASCII 주석이 있으면 설치 자체에 실패한다 (외부 검증 발견).
    # 모든 설치 진입점을 ASCII로 고정해 계급 전체를 차단한다.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in root.glob("requirements*.txt"):
        path.read_bytes().decode("ascii")  # 비ASCII면 UnicodeDecodeError로 실패


# ── Console discipline: a message the operator must read must be printable ──

def _non_docstring_literals(
        tree: ast.AST,
        non_console_tables: frozenset[str] = frozenset(),
) -> list[tuple[int, str]]:
    """Every string literal except docstrings.

    Docstrings never reach a console, so they may hold typographic characters.
    Anything else might be raised, printed, or formatted into a message.
    """
    ignored = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ignored.add(id(body[0].value))
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id in non_console_tables
                        for target in node.targets)):
            ignored.update(id(value) for value in ast.walk(node.value)
                           if isinstance(value, ast.Constant)
                           and isinstance(value.value, str))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in ignored]


def test_non_console_table_exemption_does_not_hide_prints():
    tree = ast.parse('VERDICT_GLYPH = {"PASS": "✓"}\nprint("✓")')
    literals = _non_docstring_literals(tree, frozenset({"VERDICT_GLYPH"}))
    assert literals == [(2, "✓")]


def test_operator_messages_survive_a_cp949_console():
    # A Windows Korean console raises UnicodeEncodeError on characters CP949
    # cannot encode, so an em dash in an error message crashes the tool at the
    # exact moment the operator needs to read it. This project has already been
    # bitten once; the rule is enforced rather than remembered.
    offenders = []
    for src in _sources():
        tree = ast.parse(src.read_text(encoding="utf-8"))
        # Qt renders these glyphs; they never touch a Windows console. Keep the
        # exemption scoped to this one table so a print in the same module is
        # still rejected (covered above).
        non_console_tables = (frozenset({"VERDICT_GLYPH"})
                              if src.name == "palette.py" else frozenset())
        for line, text in _non_docstring_literals(tree, non_console_tables):
            bad = sorted({c for c in text
                          if ord(c) > 127 and not c.encode("cp949", "ignore")})
            if bad:
                offenders.append(f"{src.name}:{line} {bad}")
    assert not offenders, (
        "CP949 콘솔에서 죽는 문자가 메시지에 있다:\n  " + "\n  ".join(offenders)
    )
