# Course Neurocomputing

Source for the course neurocomputing taught at the Chemnitz University of Technology.

<https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing>

## Installation 

The book is created using Quarto:

<https://quarto.org/>

Just install it and type `make slides` or `make notes` depending on whether you want to compile the slides or the book. The result is in `docs/`. 

## Usage

Shortcuts for the reveal presentations:

* `q`: shows pointer.
* `c`: shows pen.
* `s`: shows presenter mode.
* `e`: pdf printing.
* `f`: fullscreen.

## Printing to pdf

Export to pdf works better using Decktape and a huge slide size (<https://github.com/astefanutti/decktape/issues/151>)

```bash
npm install -g decktape
```

The script `generate_pdf.sh` takes the html slide as an input and generates the pdf at the correct location.