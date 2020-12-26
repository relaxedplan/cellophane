import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from PartialIdentification import PartialIdentification

os.chdir(r'C:\Users\Malta\FairnessWithUnobservedProtectedClass\Warfrin')

proxy_cols = ['VKORC1..4451.', 'VKORC1.2255.', 'VKORC1.3730.', 'VKORC1.1542.', 'VKORC1.1173.', 'VKORC1.497.', 'VKORC1..1639.', 'Acetaminophen.', 'Acetaminophen.hi.dose.', 'Simvastatin.', 'Atorvastatin.', 'Fluvastatin.', 'Lovastatin.', 'Pravastatin.', 'Rosuvastatin.', 'Cerivastatin.', 'Amiodarone.', 'Carbamazepine.', 'Phenytoin.', 'Rifampin.', 'Sulfonamide.Antibiotics.', 'Macrolide.Antibiotics.', 'Anti.fungal.Azoles.', 'Herbal.Medications..Vitamins..Supplements.']
primary = pd.read_csv('primary.csv')
primary['target'] = primary['therapeut_dose'] > 35
auxiliary = pd.read_csv('auxiliary.csv')

X_cols = list(set(primary.columns).difference(('race', 'target', 'therapeut_dose')))

base_rfc = RandomForestRegressor()
base_rfc.fit(primary[X_cols], primary['therapeut_dose'])
primary['prediction'] = base_rfc.predict(primary[X_cols]) > 35


np.random.seed(42)

PI = PartialIdentification(primary,
                           auxiliary,
                           'target',
                           X_cols,
                           'race',
                           'prediction',
                           proxy_cols)

PI.generate_report()