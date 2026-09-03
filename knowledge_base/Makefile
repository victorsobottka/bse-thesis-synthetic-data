PDF = knowledge_base.pdf
MAIN = main.tex
CHAPTERS = ch01_numbers.tex ch02_exp_log.tex ch03_calculus1.tex \
           ch04_calculus2.tex ch05_linear_algebra.tex ch06_probability.tex ch07_statistics.tex \
           ch08_time_series.tex ch09_heavy_tails.tex ch13_gans.tex \
           ch15_evaluation.tex ch16_downstream.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(MAIN) $(CHAPTERS)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	mv main.pdf $(PDF)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
