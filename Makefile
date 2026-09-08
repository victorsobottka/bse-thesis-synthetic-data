.PHONY: report report-smoke verify

# Refuses if reports/production/pipeline_run_metadata.json reports
# smoke_test=true. Reads thesis_results/production/ and reports/production/
# (generate_report.py's defaults).
report:
	python3 generate_report.py

# Renders anyway, stamped (banner + watermark) as report_<DATE>_SMOKE.pdf.
# Explicitly reads thesis_results/smoke/ and reports/smoke/ -- the
# production/ defaults would be either empty or, worse, stale production
# data/metadata mismatched with a smoke_test flag.
report-smoke:
	python3 generate_report.py --allow-smoke --results-dir thesis_results/smoke --reports-dir reports/smoke

verify:
	python3 verify_notebook.py
