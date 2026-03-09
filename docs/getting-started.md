# Getting Started

## Basic usage
```python
from selfx.dash.dashboard import SelfXDash
from selfx.backend.features import Feature

class Feature1(Feature):
    def perform(self, start, end):
        return 1

    def layout(self, role, analysis, start, end):
        return "Feature 1 succeded"

    def icon(self):
        return 'home'

class Feature2(Feature):
    required_features = ["Feature1"]

    def perform(self, start, end):
        return 2

    def layout(self, role, analysis, start, end):
        return "Feature 2 succeded"

    def icon(self):
        return 'bar_chart'

app = SelfXDash()
app.add_system('System 1', features=[Feature1, Feature2])

app.run(port=8050, host="127.0.0.1")
```

