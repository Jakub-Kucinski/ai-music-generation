# Installation
```bash
python3 -m venv env
source env/bin/activate
pip install poetry
poetry install

sudo apt update
sudo apt-get install -y git-lfs
git lfs install
sudo apt install -y musescore
sudo apt-get install -y abcmidi
sudo apt-get install -y timidity
sudo apt-get install -y fluidsynth
```

# Sound fonts
Tested sound fonts were downloaded from https://sites.google.com/site/soundfonts4u/
- Dore Mark's NY S&S Model B-v5.2.sf2
- Dore Mark's Yamaha S6-v1.6.sf2
- Essential Keys-sforzando-v9.6.sf2

All sound fonts resulted in similar performance in terms of Meta's audiobox aesthetics, so I chose to use "Essential Keys-sforzando-v9.6.sf2" as it contains sound fonts for most common instruments (excluding drums).
