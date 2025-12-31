import re

# Read current README
with open('README.md', 'r') as f:
    content = f.read()

# Add badges after the title
badges_section = '''[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: M1/M2](https://img.shields.io/badge/platform-M1%2FM2%20Mac-9cf)](https://developer.apple.com/mac/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Release](https://img.shields.io/github/v/release/jamesenglis/geospatial-vit-drone)](https://github.com/jamesenglis/geospatial-vit-drone/releases)
[![GitHub Stars](https://img.shields.io/github/stars/jamesenglis/geospatial-vit-drone?style=social)](https://github.com/jamesenglis/geospatial-vit-drone/stargazers)'''

# Update README
lines = content.split('\n')
updated_lines = []
for line in lines:
    if line.startswith('# 🛸 Geospatial Vision Transformer'):
        updated_lines.append(line)
        updated_lines.append('')  # Empty line
        updated_lines.append(badges_section)
    else:
        updated_lines.append(line)

# Write back
with open('README.md', 'w') as f:
    f.write('\n'.join(updated_lines))

print("Updated README with badges")
