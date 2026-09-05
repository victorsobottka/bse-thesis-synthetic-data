.PHONY: report report-smoke verify

# Refuses if reports/pipeline_run_metadata.json reports smoke_test=true.
report:
	python3 generate_report.py

# Renders anyway, stamped (banner + watermark) as report_<DATE>_SMOKE.pdf.
report-smoke:
	python3 generate_report.py --allow-smoke

verify:
	python3 verify_notebook.py
