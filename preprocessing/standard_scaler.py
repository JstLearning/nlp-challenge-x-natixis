from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def get_column_transformer():
    ct = ColumnTransformer([
        ('Standard Scaler', StandardScaler(), [
                                        'Index - 9',
                                        'Index - 8',
                                        'Index - 7',
                                        'Index - 6',
                                        'Index - 5',
                                        'Index - 4',
                                        'Index - 3',
                                        'Index - 2',
                                        'Index - 1',
                                        'Index - 0'])
    ], remainder='passthrough')
    return ct