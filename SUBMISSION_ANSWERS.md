# Peer Review Submission Answers

1. **Are there unit tests for the API?**  
   Yes. See `tests/test_api.py`.

2. **Are there unit tests for the model?**  
   Yes. See `tests/test_model.py`.

3. **Are there unit tests for the logging?**  
   Yes. See `tests/test_logging.py`.

4. **Can all unit tests be run with a single script and do all tests pass?**  
   Yes. Run `python run_tests.py`.

5. **Is there a mechanism to monitor performance?**  
   Yes. Prediction latency is written to `logs/predictions.jsonl`. `src/monitor.py` and the `/monitor` endpoint report request count, average latency, and p95 latency.

6. **Was there an attempt to isolate read/write unit tests from production models and logs?**  
   Yes. `tests/conftest.py` overrides `MODEL_PATH` and `PREDICTION_LOG` to temporary test directories.

7. **Does the API work as expected for a specific country and all countries combined?**  
   Yes. `POST /predict` accepts a supported country name or `"all"`. Tests cover both behaviors.

8. **Does data ingestion exist as a function or script to facilitate automation?**  
   Yes. `src/data_ingestion.py` exposes `generate_business_data()` and can be run as a script.

9. **Were multiple models compared?**  
   Yes. The workflow compares a mean baseline (`DummyRegressor`), Linear Regression, Gradient Boosting, and Random Forest.

10. **Did the EDA investigation use visualizations?**  
    Yes. `src/eda.py` creates country revenue, monthly revenue, and correlation visualizations.

11. **Is everything containerized within a working Docker image?**  
    Yes. The repository includes a `Dockerfile` that installs dependencies, generates data, trains the model, and starts the API.

12. **Was a visualization used to compare the model to the baseline?**  
    Yes. `src/train.py` creates `reports/model_comparison.png`, comparing RMSE across the baseline and candidate models.
