import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import org.apache.spark.sql.functions._
import spark.implicits._
import org.apache.log4j._

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SaveMode

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header","true").option("inferSchema","true").option("multiline","true").format("json").load("/Users/pankhuri/Documents/Projects/Extras/sports/ipl_json/*.json")

data.createOrReplaceTempView("data");

val df_city = data.select($"meta.created" as "created", $"info.city" as "city", explode($"innings")).withColumn("team", $"col.team").select($"*", explode($"col.overs") as "overr_s").select($"overr_s.over" as "over_id", $"*", explode($"overr_s.deliveries") as "delivery").withColumn("runs", $"delivery.runs.total").drop($"col").drop($"delivery").drop("overr_s").distinct()

// Extract relevant data from ipl_json. Fields: created,city, over_id, team, sum(runs)
val filtered_rows_city = df_city.where($"over_id" === 5 || $"over_id" === 9 || $"over_id" === 14 || $"over_id" === 19)
val result_city = filtered_rows_city.groupBy("created", "city", "over_id","team").agg(sum("runs"))
result_city.write.mode(SaveMode.Overwrite).option("header",true).csv("/Users/pankhuri/Documents/Projects/Extras/sports/mined_data")


// Get list of all teams
val teams_list = result_city.select($"team").distinct()

// Convert the extracted data to only have fields usable in regression for a particular team: over_id, sum(runs)
val team_df = result_city.where($"team" === "Mumbai Indians").select($"over_id", $"sum(runs)" as "total_runs")

// Apply linear regression
val df_l1 = team_df.select($"total_runs".as("label"),$"over_id")
val assembler = new VectorAssembler().setInputCols(Array("over_id")).setOutputCol("features")
val output = assembler.transform(df_l1).select($"label",$"features")

// Split data to have 70% training data and 30% test data
val Array(training, test) = output.select("label","features").randomSplit(Array(0.7, 0.3), seed = 12345)
val lr = new LinearRegression()
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,Array(1000,0.001)).build()
val trainValidationSplit = (new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator().setMetricName("r2")).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8))
val model = trainValidationSplit.fit(training)
model.transform(test).select("features", "label", "prediction").show()