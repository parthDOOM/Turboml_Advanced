feature-pipeline:
	python setup_feature_pipeline.py

model:
	python setup_model.py

live-data:
	python generate_live_data.py

all: feature-pipeline model live-data

check-turbo-ml-installation:
	@echo "Checking import turboml_installer works"
	python -c "import turboml_installer"

	@echo "Checking import turboml works"
	python -c "import turboml"