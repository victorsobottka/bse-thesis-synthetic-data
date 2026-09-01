PDF = knowledge_base.pdf
MAIN = main.tex
CHAPTERS = ch1_foundations.tex ch2_time_series.tex ch3_heavy_tails.tex \
           ch4_gans.tex ch5_evaluation.tex ch6_downstream.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(MAIN) $(CHAPTERS)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	mv main.pdf $(PDF)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
