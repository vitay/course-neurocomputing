import sys
import re

filename = sys.argv[1]

with open(filename, 'r') as f:
    data = f.readlines()

output = []

for line in data:
    line = line.replace("[leftcol]", "::: {.columns}\n::: {.column width=50%}\n")
    line = line.replace("[rightcol]", ":::\n::: {.column width=50%}\n")
    line = line.replace("[endcol]", ":::\n:::\n")

    p = re.compile(r'\[leftcol (\d\d)\]')
    match = p.match(line)
    if match is not None:
        line = p.sub(r"::: {.columns}\n::: {.column width=\1%}", line)

    p = re.compile(r'\[rightcol (\d\d)\]')
    match = p.match(line)
    if match is not None:
        line = p.sub(r":::\n::: {.column width=\1%}", line)

    p = re.compile(r'\[citation (.*)\]')
    match = p.match(line)
    if match is not None:
        line = p.sub(r"::: footer\n\1\n:::", line)

    output.append(line)


with open(filename, 'w') as f:
    f.writelines(output)