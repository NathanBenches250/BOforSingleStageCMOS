# single CMOS OpAmp Optimization using Bayesian Optimization

## Project Structure

- `amp_template.net`: LTSpice netlist template for the single-stage OpAmp
- `automate.py`: Utilities for automating LTSpice simulations and result extraction
- `BO.py`: Standard Bayesian Optimization implementation using scikit-optimize (skopt)
- `surrogate_model.py`: Surrogate models to predict circuit performance
- `custom_bo.py`: Custom Bayesian Optimization implementation using our trained surrogate models
- `requirements.txt`: Required Python packages for this project

## Folder Structure
```
CSCI5980PROJECT\
  | - code\ 
      | - automate.py
      | - BO.py
      | - clean.py      
      | - custom_bo.py      
      | - surrogate_model.py
   | - csv_outputs\ "data collected from LTspice simulations
   | - results\ "graphs comparing the approaches"
   | - sources\ "sources for some decisions in the project"

```

### Performance Comparison:
- Random sampling best score: -34.7783
- Standard BO (skopt) best score: -33.6385
- Custom BO best score: -32.8988

## Running the Code

```
# Install dependencies
pip install -r requirements.txt

# Run standard BO approach
python BO.py

# Run custom BO approach
python custom_bo.py
```
