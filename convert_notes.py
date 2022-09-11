import sys
import re

filename = sys.argv[1]

with open(filename, 'r') as f:
    data = f.read()

print(data)

print("-"*60)

# Figures
p = re.compile(r"```{figure} ../img/(.*?)\n(.*?)---\nwidth: (\d+)%\n---\n(.*?)\n```", re.DOTALL)
match = p.search(data)
print(match)

if match is not None:
    data = p.sub(r"![\4](../slides/img/\1){width=\3%}", data)

print(data)

print("-"*60)

# Youtube
p = re.compile(r"(.*)https://www.youtube.com/embed/(.*?)'(.*)")
match = p.search(data)
print(match)

if match is not None:
    data = p.sub(r"{{< youtube  \2 >}}", data)

# Links
p = re.compile(r"(.*){cite}`(.*?)`(.*)")
match = p.search(data)
print(match)

if match is not None:
    data = p.sub(r"\1[@\2]\3", data)

# Notes
p = re.compile(r"```{note}\n(.*?)\n```", re.DOTALL)
match = p.search(data)
print(match)

if match is not None:
    data = p.sub(r":::{.callout-note}\n\1\n:::", data)


with open(filename, 'w') as f:
    f.write(data)