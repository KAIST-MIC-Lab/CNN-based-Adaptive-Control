# MIC LaTeX Template

## About This Template

This template is made for following objectives:

- Esay start to write draft with LaTex.
- Encourage lab members to use same notations.
- Fast coding with simple snippets.

## How to use

Add this template using the following line.
```bash
$ git submodule add https://github.com/KAIST-MIC-Lab/template   
```

Initialize and update the lastest template.
```bash
$ git submodule init
$ git submodule update
```

Insert the following line in your LaTex manuscript.
```tex
\def\pub{false} % true for publication, false for draft
\newcommand*{\template}{dding_template}
\input{\template/preamble/preamble_conf.tex}
% \input{\template/preamble/preamble_article.tex}
````

If you choose `true` argument for `\pub`, the hyperlinks will be colored.
Every package, macro and symbol are defined in preamble files.
Select as the type of your manuscript.

## Table of Contents

This template includes...
```
.
├── IEEE
├── README.md
├── macros
│   ├── macros_general.tex
│   └── macros_math.tex
├── packages
│   └── packages_general.tex
├── preamble
│   ├── preamble_article.tex
│   └── preamble_conf.tex
├── refs.bib
└── symbols
    ├── symbols_NN.tex
    ├── symbols_motor.tex
    └── symbols_robot.tex
```

## Authors

- Ryu Myeongseok [@DDingR](https://github.com/DDingR)
