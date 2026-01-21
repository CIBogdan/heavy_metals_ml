import pandas as pd

from heavymetals.preprocessing import FeatureSpec, build_preprocessor


def test_preprocessor_fit_transform():
    df = pd.DataFrame(
        {
            "Metal": ["Pb", "Pb", "Cd"],
            "Origin": ["CN", "RO", "CN"],
            "x1": [1.0, 2.0, 3.0],
        }
    )
    spec = FeatureSpec(categorical=["Metal", "Origin"], numeric=["x1"])
    prep = build_preprocessor(spec)
    X = prep.fit_transform(df[spec.categorical + spec.numeric])
    assert X.shape[0] == 3
