#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper.docx and supplementary.docx from the markdown files with pandoc.

Display equations become native Word equations (OMML, via LaTeX); inline notation such as Δ_{M0,k}, ψ_L or *s*_min
becomes real subscripts and superscripts; the four figures are inserted above their captions. Styles come from a
reference document: A4, Times New Roman, black headings, bordered tables.
Run from paper/:  python3 build_docx.py [paper|supplementary|all]   ->  paper.docx, supplementary.docx
"""
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

DISPLAY = {
    "ψ_k = E[logit *p*(*X̃*) | *t* = 1] − E[logit *p*(*X̃*) | *t* = 0].":
        r"$$\psi_k = \text{E}\left[\text{logit}\, p(\tilde{X}) \mid t = 1\right] - \text{E}\left[\text{logit}\, p(\tilde{X}) \mid t = 0\right].$$",
    "logit P(*Y* = 1 | *x̃*, *R* = 1) = logit P(*Y* = 1 | *x̃*) + log *r*₁(*x̃*) − log *r*₀(*x̃*).":
        r"$$\text{logit}\, P(Y = 1 \mid \tilde{x}, R = 1) = \text{logit}\, P(Y = 1 \mid \tilde{x}) + \log r_1(\tilde{x}) - \log r_0(\tilde{x}).$$",
    "Δ_{M0,k} − Δ_{M2,k} = E[log *g*(*X̃*) | *t* = 1] − E[log *g*(*X̃*) | *t* = 0].":
        r"$$\Delta_{M0,k} - \Delta_{M2,k} = \text{E}\left[\log g(\tilde{X}) \mid t = 1\right] - \text{E}\left[\log g(\tilde{X}) \mid t = 0\right].$$",
}
WIDTHS = {8: [11, 14, 9, 15, 15, 15, 10, 11], 3: [22, 33, 45], 6: [11, 24, 13, 23, 15, 14]}   # relative column widths
FIGS = {"1": "fig1_mechanism.png", "2": "fig2_learned_reversal.png", "3": "fig3_intervention_bridge.png", "4": "fig4_external_boundary.png",
        "S1": "figS1_phase_diagram.png", "S2": "figS2_finetune.png", "S3": "figS3_logit_benchmark.png", "S4": "figS4_icdl.png"}
DOCS = {"paper": ("paper.md", "paper.docx", True), "supplementary": ("supplementary.md", "supplementary.docx", False)}

# a subscript follows an italic letter (*a*), a Greek letter, ĝ, E, OR or RR
SUB = re.compile(r"(\*[A-Za-z]\*|[ΔψπκμτσθE]|ĝ|OR|RR)_(\{[^}]*\}|[A-Za-z0-9ηω]+)")
# plain-text symbols that need italics and a subscript
PLAIN = [(re.compile(r"τ̂_k"), "τ̂~k~"), (re.compile(r"\*z\*_k,i"), "*z*~k,i~"), (re.compile(r"π_η,i"), "π~η,i~"),
         (re.compile(r"θ\*_m\b"), "θ\\*~m~"), (re.compile(r"(?<![\w*`])s_(low|high)\b"), r"*s*~\1~"),
         (re.compile(r"(?<![\w*`])z_k\b"), "*z*~k~"), (re.compile(r"(?<![\w*`])Q_\{([^}]*)\}"), r"*Q*~\1~"),
         (re.compile(r"(?<![\w*`])Q_a\b"), "*Q*~a~")]


def sub_text(body):
    body = body.strip("{}").replace("\\|", "|")
    return body.replace("\\", "\\\\").replace(" ", "\\ ").replace("|", "\\|")


def auto_widths(rows):
    """Relative widths from the longest cell text of each column, capped so that no column dominates."""
    cols = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
    n = max(len(c) for c in cols)
    size, word = [0] * n, [0] * n
    for r in cols:
        for i, c in enumerate(r):
            c = re.sub(r"[*`\\]", "", c)
            size[i] = max(size[i], min(len(c), 40))
            word[i] = max([word[i]] + [len(x) for x in c.split()])
    floor = [max(word[i] + 2, 5) for i in range(n)]            # never narrower than the longest word
    budget = 96 - 2.5 * n                                     # 9-point characters per line, less cell padding
    extra = [max(0, 0.6 * size[i] - floor[i]) for i in range(n)]
    room = max(0, budget - sum(floor))
    tot = sum(extra) or 1
    return [round(floor[i] + room * extra[i] / tot) for i in range(n)]


def widths(sep, w=None):
    cells = sep.strip().strip("|").split("|")
    w = w or WIDTHS.get(len(cells))
    if not w or len(w) != len(cells):
        return sep
    out = []
    for c, n in zip(cells, w):
        c = c.strip()
        out.append((":" if c.startswith(":") else "") + "-" * (2 * n) + (":" if c.endswith(":") else ""))
    return "| " + " | ".join(out) + " |"


def math_segment(s):
    s = s.replace("π_t^d", "π~t~^d^")
    s = re.sub(r"(?:\*s\*|(?<![\w*])s)∗", "*s*^\\*^", s)
    for pat, rep in PLAIN:
        s = pat.sub(rep, s)
    return SUB.sub(lambda x: f"{x.group(1)}~{sub_text(x.group(2))}~", s)


def convert(md, fixed_widths=True):
    out = []
    lines = md.split("\n")
    for k, line in enumerate(lines):
        if re.match(r"^\|\s*:?-{3,}", line):
            w = None
            if not fixed_widths:
                j = k + 1
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                w = auto_widths([lines[k - 1]] + lines[k + 1:j])
            out.append(widths(line, w))
            continue
        if line.strip() in DISPLAY:
            out.append(DISPLAY[line.strip()])
            continue
        m = re.match(r"^\*\*Fig\. (S?\d)\.\*\*", line)
        if m:
            out.append(f"![](figures/{FIGS[m.group(1)]}){{width=6.2in}}\n")
        if not line.startswith("#"):
            parts = line.split("`")                      # leave code spans such as file names untouched
            line = "`".join(math_segment(p) if i % 2 == 0 else p for i, p in enumerate(parts))
        out.append(line)
    return "\n".join(out)


def reference_doc(path):
    src = path + ".src"
    subprocess.run(["pandoc", "-o", src, "--print-default-data-file", "reference.docx"], check=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                x = data.decode("utf-8")
                x = re.sub(r'<w:rFonts [^>]*/>', '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>', x)
                x = re.sub(r'<w:color [^>]*w:themeColor="(?:accent1|text1)"[^>]*/>', '<w:color w:val="000000"/>', x)
                x = x.replace("</w:styles>", '<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText"><w:name w:val="Table Text"/>'
                              '<w:basedOn w:val="Compact"/><w:qFormat/><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                              '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style></w:styles>')
                x = x.replace("<w:sz w:val=\"24\"/>", "<w:sz w:val=\"22\"/>")
                x = re.sub(r'(<w:style [^>]*w:styleId="VerbatimChar"[^>]*>.*?<w:rPr>).*?(</w:rPr>)',
                           r'\1<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="18"/><w:szCs w:val="18"/>\2', x, count=1, flags=re.S)
                x = re.sub(r"<w:tblCellMar>.*?</w:tblCellMar>", lambda m: m.group(0).replace('"108"', '"57"'), x, count=1, flags=re.S)
                # bordered tables
                x = re.sub(r'(<w:style w:[^>]*w:styleId="Table"[^>]*>.*?<w:tblPr>.*?<w:tblInd [^>]*/>)',
                           r'\1<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                           r'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                           r'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders>',
                           x, count=1, flags=re.S)
                data = x.encode("utf-8")
            if item.filename == "word/theme/theme1.xml":
                data = re.sub(rb'typeface="Aptos[^"]*"', b'typeface="Times New Roman"', data)
            if item.filename == "word/document.xml":
                x = data.decode("utf-8")   # A4 with 2.5 cm margins
                x = re.sub(r"<w:pgSz[^>]*/>", '<w:pgSz w:w="11906" w:h="16838"/>', x)
                x = re.sub(r"<w:pgMar[^>]*/>", '<w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/>', x)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    os.remove(src)


def build(name):
    src_md, out_name, fixed = DOCS[name]
    md = convert(open(os.path.join(HERE, src_md), encoding="utf-8").read(), fixed)
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "doc.md")
    open(src, "w", encoding="utf-8").write(md)
    ref = os.path.join(tmp, "reference.docx")
    reference_doc(ref)
    out = os.path.join(HERE, out_name)
    subprocess.run(["pandoc", src, "-f", "markdown+pipe_tables+subscript+superscript+tex_math_dollars", "-t", "docx",
                    "--reference-doc", ref, "--shift-heading-level-by=-1", "--resource-path", HERE, "-o", out], check=True)
    shutil.rmtree(tmp)
    table_text(out)
    print("built", out)


def table_text(path):
    """Use the 9-point Table Text paragraph style inside tables."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml" and b'Extension="png"' not in data:
                data = data.replace(b"<Types ", b"<Types ", 1).replace(b">", b'><Default Extension="png" ContentType="image/png"/>', 1) \
                    if data.startswith(b"<Types") else data.replace(b"<Default ", b'<Default Extension="png" ContentType="image/png"/><Default ', 1)
            if item.filename == "word/document.xml":
                x = data.decode("utf-8")
                x = x.replace('<m:nor /><m:sty m:val="p" />', "<m:nor />")      # nor and sty are alternatives in OMML
                x = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: m.group(0).replace('<w:pStyle w:val="Compact" />', '<w:pStyle w:val="TableText" />'), x, flags=re.S)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    os.replace(tmp, path)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (DOCS if which == "all" else [which]):
        build(name)
