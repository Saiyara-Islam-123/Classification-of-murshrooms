import pandas as pd
df = pd.read_csv("mushrooms.csv")
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

le_class = LabelEncoder() #done
le_cap_shape = LabelEncoder() #done
le_cap_surface = LabelEncoder() #done
le_cap_color = LabelEncoder() #done
le_bruises = LabelEncoder() #done
le_odor = LabelEncoder()#done
le_gill_attachment = LabelEncoder()#done
le_gill_spacing = LabelEncoder()#done
le_gill_size = LabelEncoder()#done
le_gill_color = LabelEncoder()#done
le_stalk_shape = LabelEncoder()#done
le_stalk_root = LabelEncoder() #done
le_stalk_surface_above_ring = LabelEncoder()#done
le_stalk_surface_below_ring = LabelEncoder()#done
le_stalk_color_above_ring = LabelEncoder()#done
le_stalk_color_below_ring = LabelEncoder()#done
le_veil_type = LabelEncoder()#done
le_veil_color = LabelEncoder()#done
le_ring_number = LabelEncoder()#done
le_spore_print_color = LabelEncoder() #done

df["class_n"] = le_class.fit_transform(df["class"])

df["cap_shape_n"] = le_cap_shape.fit_transform(df["cap-shape"])
df["cap_surface_n"] = le_cap_surface.fit_transform(df["cap-surface"])
df["cap_color_n"] = le_cap_color.fit_transform(df["cap-color"])
df["bruises_n"] = le_bruises.fit_transform(df["bruises"])
df["odor"] = le_odor.fit_transform(df["odor"])
df["gill_attachement_n"] = le_gill_attachment.fit_transform(df["gill-attachment"])
df["gill_spacing_n"] = le_gill_spacing.fit_transform(df["gill-spacing"])
df["gill_size_n"] = le_gill_size.fit_transform(df["gill-size"])
df["gill_color_n"] = le_gill_color.fit_transform(df["gill-color"])
df["stalk_shape_n"] = le_stalk_shape.fit_transform(df["stalk-shape"])
df["stalk_root_n"] = le_stalk_root.fit_transform(df["stalk-root"])
df["stalk_surface_above_ring_n"] = le_stalk_surface_above_ring.fit_transform(df["stalk-surface-above-ring"])
df["stalk_surface_below_ring_n"] = le_stalk_surface_below_ring.fit_transform(df["stalk-surface-below-ring"])
df["stalk_color_above_ring_n"] = le_stalk_color_above_ring.fit_transform(df["stalk-color-above-ring"])
df["stalk_color_below_ring_n"] = le_stalk_color_below_ring.fit_transform(df["stalk-color-below-ring"])
df["veil_type_n"] = le_veil_type.fit_transform(df["veil-type"])
df["veil_color_n"] = le_veil_color.fit_transform(df["veil-color"])
df["ring_number_n"] = le_ring_number.fit_transform(df["ring-number"])
df["spore_print_color_n"] = le_spore_print_color.fit_transform(df["spore-print-color"])

df_n = df.drop(["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"], axis = "columns")


train, validation, test = np.split(df_n.sample(frac=1, random_state = 42), [int(0.6*len(df_n)),
int(0.8*len(df_n))])

X_train = train.drop(["class_n"], axis = "columns")
X_test = test.drop(["class_n"], axis = "columns")
X_validation = validation.drop(["class_n"], axis = "columns")

y_train = train["class_n"]
y_test = test["class_n"]
y_validation = validation["class_n"]

model = RandomForestClassifier()
model.fit(X_train, y_train)
