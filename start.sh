#!/bin/bash
python app.py &
python -m http.server $PORT