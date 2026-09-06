.PHONY: report report-smoke verify

# Refuses if reports/pipeline_run_metadata.json reports smoke_test=true.
# Reads thesis_results/production/ (generate_report.py's default).
report:
	python3 generate_report.py

# Renders anyway, stamped (banner + watermark) as report_<DATE>_SMOKE.pdf.
# Explicitly reads thesis_results/smoke/ -- the production/ default would be
# either empty or, worse, stale production data mismatched with a smoke_test
# metadata flag.
report-smoke:
	python3 generate_report.py --allow-smoke --results-dir thesis_results/smoke

verify:
	python3 verify_notebook.py
