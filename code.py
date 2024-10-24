from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssem, PCA
from pyspark.ml.stat import Correlation, ChiSquareTest

# Initialize Spark session
s = SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Load the dataset from the extracted CSV file path
data = s.read.csv("/content/ai4i2020.csv", header=True, inferSchema=True)  # Update the path

# Display basic info
data.printSchema()

# Check for missing values
miss_val = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
miss_val.show()

# Handle missing values (example: fill with mean for numeric columns)
# Retrieve numeric columns from the schema
num_columns = [field.name for field in data.schema.fields if field.dataType.typeName() in ['integer', 'double']]
for column_n in num_columns:
    mean_val = data.agg({column_n: "mean"}).first()[0]
    data = data.na.fill({column_n: mean_val})

# Binning continuous variable (use 'Air temperature [K]' or any numeric column)
split= [-float("inf"), 300, 320, 340, float("inf")]  # Update split according to your requirements
vectorizer = Bucketizer(split=split, ipCol="Air temperature [K]", opCol="temp_binned")  # Change to a relevant column
data = bucketizer.transform(data)

# Index categorical columns (update 'Machine fail' with the correct categorical column)
indexer = StringIndexer(ipCol="Machine fail", opCol="machine_fail_index")  # Replace 'Machine fail' with the actual column name
data = indexer.fit(data).transform(data)

# One-Hot Encoding
encoder = OneHotEncoder(ipCol="machine_fail_index", opCol="machine_fail_ohe")
data = encoder.fit(data).transform(data)

# Check for duplicate rows
dup = data.groupBy(data.columns).count().filter("count > 1")
dup.show()

# Correlation matrix (example: for numerical columns)
vector_col = "corr_features"
assemb = VectorAssem(ipCols=num_columns, opCol=vector_col)
dataframe_vector = assem.transform(data).select(vector_col)

correlation_matrix = Correlation.corr(df_vector, vector_col).head()[0]
print(f"Pearson correlation matrix:\n{correlation_matrix}")

# Chi-Square test (for nominal features and the target variable)
# Update 'feature_column' with the actual feature name to be tested
chi_sq_test = ChiSquareTest.test(data, featuresCol="machine_fail_ohe", label_Column="machine_fail_index")  # Update 'feature_column'
chi_sq_test.show()

# Prepare feature vector for PCA
assem = VectorAssem(ipCols=num_columns, opCol="features")  # Update with numeric features
data_feat = assem.transform(data)

# Apply PCA
Principal_comp = PCA(k=3, ipCol="features", opCol="pca_feat")  # Set k as per requirement
Principal_compModel = pca.fit(data_feat)
principalcomp_result = principal_compmodel.transform(data_feat)

# Show the result of PCA
principalcomp_result.select("pca_feat").show(truncate=False)

# Stop Spark session
spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssem, PCA
from pyspark.ml.stat import Correlation, ChiSquareTest

# Initialize Spark session
spark = SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Load the dataset
data = spark.read.csv("ai4i2020.csv", header=True, inferSchema=True)

# Display basic info
data.printSchema()

# Check for missing values
miss_val = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
miss_val.show()

# Handle missing values (example: fill with mean for numeric columns)
num_columns = [c for c, dtype in data.dtypes.items() if dtype in ['int', 'double']]
for column_n in num_columns:
    mean_val = data.agg({column_n: "mean"}).first()[0]
    data = data.na.fill({column_n: mean_val})

# Binning continuous variable (example: 'age' column)
split = [-float("inf"), 20, 30, 40, float("inf")]
bucketizer = Bucketizer(split=split, ipCol="age", opCol="age_binned")
data = bucketizer.transform(data)

# Index categorical columns (example: 'fail')
indexer = StringIndexer(ipCol="fail", opCol="fail_index")
data = indexer.fit(data).transform(data)

# One-Hot Encoding
encoder = OneHotEncoder(ipCol="fail_index", opCol="fail_ohe")
data = encoder.fit(data).transform(data)

# Check for duplicate rows
dup = data.groupBy(data.columns).count().filter("count > 1")
dup.show()

# Correlation matrix (example: for numerical columns)
vector_col = "corr_features"
assem = VectorAssem(ipCols=num_columns, opCol=vector_col)
dataframe_vector = assem.transform(data).select(vector_col)

correlation_matrix = Correlation.corr(df_vector, vector_col).head()[0]
print(f"Pearson correlation matrix:\n{correlation_matrix}")

# Chi-Square test (for nominal features and the target variable)
chi_sq_test = ChiSquareTest.test(data, featuresCol="feature_column", label_Column="fail_index")  # Update 'feature_column'
chi_sq_test.show()

# Prepare feature vector for PCA
assem = VectorAssem(ipCols=num_columns, opCol="features")  # Update with numeric features
data_feat = assem.transform(data)

# Apply PCA
principal_comp = PCA(k=3, ipCol="features", opCol="principal_compFeat")  # Set k as per requirement
principal_compModel = pca.fit(data_feat)
principalcomp_result = principal_compmodel.transform(data_pca_feat)

# Show the result of PCA
principalcomp_result.select("principal_compFeat ").show(truncate=False)

# Stop Spark session
spark.stop()
