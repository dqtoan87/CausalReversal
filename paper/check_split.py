# -*- coding: utf-8 -*-
"""Consistency checker for a (paper, supplementary) pair. Usage: check_split.py paper1.md supplementary1.md"""
import io, re, sys, math, collections, os

def load(p):
    return io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), p), encoding='utf-8').read()

def tables_ok(s, tag):
    L = [l.replace('\\|', '\u2758') for l in s.split('\n')]
    issues, i = [], 0
    while i < len(L):
        if L[i].startswith('|') and i + 1 < len(L) and re.match(r'^\|[\s:\-|]+\|$', L[i + 1]):
            h = L[i].count('|')
            if L[i + 1].count('|') != h:
                issues.append('%s TABLE line %d header %d vs separator %d' % (tag, i + 1, h - 1, L[i + 1].count('|') - 1))
            j = i + 2
            while j < len(L) and L[j].startswith('|'):
                if L[j].count('|') != h:
                    issues.append('%s TABLE line %d has %d cells, header %d' % (tag, j + 1, L[j].count('|') - 1, h - 1))
                j += 1
            i = j
        else:
            i += 1
    return issues

def main(pp, sp):
    paper, supp = load(pp), load(sp)
    issues = []
    issues += tables_ok(paper, 'PAPER')
    issues += tables_ok(supp, 'SUPP')

    # main-text sections
    secs = set(re.findall(r'^### ([A-G])\.', paper, re.M)) | set(re.findall(r'^## ([IVX]+)\.', paper, re.M))
    for m in re.finditer(r'Section ([IVX]+)-([A-G])', paper + supp):
        if m.group(2) not in secs:
            issues.append('XREF Section %s-%s missing' % m.groups())
    # main-text section letters used bare inside supplementary tables
    for m in set(re.findall(r'\| (III-[A-G]|IV-[A-G])[,\s|]', supp)):
        if m.split('-')[1] not in secs:
            issues.append('XREF bare %s missing' % m)

    # main tables and figures
    tabs = set(re.findall(r'^\*\*TABLE (\d+)\.', paper, re.M))
    for m in set(re.findall(r'Table (\d+)\b', paper + supp)):
        if m not in tabs:
            issues.append('XREF Table %s referenced but not defined' % m)
    for t in tabs:
        if not re.search(r'Table %s\b' % t, paper):
            issues.append('XREF Table %s defined but never referenced in text' % t)
    figs = set(re.findall(r'^\*\*Fig\.? (\d)\.\*\*', paper, re.M))
    for m in set(re.findall(r'Fig\. (\d)', paper)):
        if m not in figs:
            issues.append('XREF Fig %s referenced but not defined' % m)
    for f in figs:
        if len(re.findall(r'Fig\. %s\b' % f, paper)) < 2:
            issues.append('XREF Fig %s defined but never referenced in text' % f)

    # supplementary sections and tables
    sdef = set(int(x) for x in re.findall(r'^## S(\d+)\.', supp, re.M))
    cited = lambda t: set(int(x) for g in re.findall(r'Sections? (S\d+(?:(?:, | and )S\d+)*)', t) for x in re.findall(r'S(\d+)', g))
    for m in cited(paper + supp):
        if m not in sdef:
            issues.append('XREF Supplementary Section S%d missing' % m)
    for d in sorted(sdef):
        if d not in cited(paper):
            issues.append('XREF Supplementary S%d never cited from the paper' % d)
    stab = [int(x) for x in re.findall(r'^\*\*TABLE S(\d+)\.', supp, re.M)]
    if stab != list(range(1, len(stab) + 1)):
        issues.append('SUPPTAB labels not sequential: %s' % stab)
    for m in set(int(x) for x in re.findall(r'Table S(\d+)', supp + paper)):
        if m not in stab:
            issues.append('XREF Table S%d referenced but not defined' % m)
    for t in stab:
        if not re.search(r'Table S%d\b' % t, supp + paper):
            issues.append('XREF Table S%d defined but never referenced' % t)

    # references
    defined = set(int(x) for x in re.findall(r'^\[(\d+)\] \w', paper, re.M))
    body = paper.split('## References')[0]
    cited = set(int(x) for x in re.findall(r'\[(\d+)\]', body))
    if cited - defined:
        issues.append('REF cited but undefined: %s' % sorted(cited - defined))
    if defined - cited:
        issues.append('REF defined but uncited: %s' % sorted(defined - cited))
    order = []
    for m in re.finditer(r'\[(\d+)\]', body):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
    if order != sorted(order):
        k = next(i for i, (a, b) in enumerate(zip(order, sorted(order))) if a != b)
        issues.append('REF citations not in order of first appearance (position %d: %s vs %s)'
                      % (k, order[k], sorted(order)[k]))

    # E-value arithmetic in any table row of the form | label | +x | [ci] | rr | evalue |
    for m in re.finditer(r'^\|\s*([^|]+?)\s*\|\s*([+\-−][^|]*)\|\s*\[[^\]]*\]\S*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', paper, re.M):
        try:
            rr, ev = float(m.group(3)), float(m.group(4))
        except ValueError:
            continue
        if rr <= 0:
            continue
        r = rr if rr >= 1 else 1 / rr
        exp = r + math.sqrt(r * (r - 1))
        if abs(exp - ev) > 0.02:
            issues.append('EVALUE %s RR=%.2f E=%.2f expected %.2f' % (m.group(1)[:28], rr, ev, exp))

    # spelling consistency
    for a, b in [('colour', 'color'), ('artefact', 'artifact'), ('modelling', 'modeling'),
                 ('modelled', 'modeled'), ('behaviour', 'behavior'), ('licence', 'license')]:
        na = len(re.findall(r'\b%s\b' % a, body, re.I))
        nb = len(re.findall(r'\b%s\b' % b, body, re.I))
        if na and nb:
            issues.append('SPELL %s=%d vs %s=%d mixed' % (a, na, b, nb))

    # duplicate sentences
    nd = re.sub(r'\|[^\n]*\n', '', body)
    sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', nd) if len(x.strip()) > 70]
    c = collections.Counter(re.sub(r'[^a-z ]', '', x.lower()) for x in sents)
    for k, v in c.items():
        if v > 1:
            issues.append('DUP sentence repeated %dx: %s' % (v, k[:80]))

    # dash policy
    for tag, txt in (('PAPER', body), ('SUPP', supp)):
        for m in re.finditer(r'[^\d\s]\s*[—–]\s*[^\d\s]', txt):
            issues.append('%s DASH near: %s' % (tag, txt[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')))

    abst = len(paper.split('## Abstract', 1)[1].strip().split('\n\n', 1)[0].split()) if '## Abstract' in paper else 0
    tbl = sum(len(l.split()) for l in body.split('\n') if l.startswith('|'))
    print('%s: WORDS %d (body %d, tables %d, text %d) | ABSTRACT %d | SUPP WORDS %d'
          % (pp, len(paper.split()), len(body.split()), tbl, len(body.split()) - tbl, abst, len(supp.split())))
    print()
    if not issues:
        print('NO ISSUES FOUND')
    for i in issues:
        print(i)
    print('\nTOTAL ISSUES:', len(issues))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
