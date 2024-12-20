# Installation

## Prerequisites
- Python 3.8+
- pip package manager

## Install via pip
```bash
pip install nosa-autostreamlit
```

## Install from GitHub

```bash
pip install git+https://github.com/thesnak/nosa-autostreamlit.git
```

## Local Development Setup

```bash
# Clone the repository
git clone https://github.com/thesnak/nosa-autostreamlit.git
cd nosa-autostreamlit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```python
import nosa_autostreamlit
print(nosa_autostreamlit.__version__)
```