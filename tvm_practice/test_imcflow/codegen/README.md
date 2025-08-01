# Running MicroTVM Models

This directory contains compiled microTVM projects that can run on x86 CPU.

## Available Models

- **small_ref**: Reference model (standard TVM compilation)
- **small_evl**: Evaluation model (IMCFLOW compilation)

## How to Run Models

### Using the Updated Script

The easiest way to run models is using the updated `run_with_tvmc.py` script:

```bash
# Run small_ref model with default settings
python run_with_tvmc.py small_ref

# Run with different input fill modes
python run_with_tvmc.py small_ref --fill-mode random
python run_with_tvmc.py small_ref --fill-mode ones
python run_with_tvmc.py small_ref --fill-mode zeros

# Customize output
python run_with_tvmc.py small_ref --print-top 3
python run_with_tvmc.py small_ref --print-top 10

# Combine options
python run_with_tvmc.py small_ref --fill-mode ones --print-top 3

# Get help
python run_with_tvmc.py --help
```

### Direct TVMC Command

You can also run models directly with TVMC:

```bash
python -m tvm.driver.tvmc run \
    --device micro \
    small_ref/micro/project \
    --fill-mode random \
    --print-top 5
```

## Expected Output

When running successfully, you should see output similar to:

```
[[2.         1.         8.         4.         0.        ]
 [0.18700646 0.14140563 0.11537166 0.1080846  0.10533262]]
```

The first row shows the top indices, and the second row shows the corresponding probabilities.