import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

import spark.implicits._
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("multiline","true").format("json").load("/Users/pankhuri/Documents/Projects/Extras/sports/ipl_json/*.json")
data.createOrReplaceTempView("data");

val df_city = data.select($"meta.created" as "created", $"info.city" as "city", explode($"innings")).withColumn("team", $"col.team").select($"*", explode($"col.overs") as "overr_s").select($"overr_s.over" as "over_id", $"*", explode($"overr_s.deliveries") as "delivery").withColumn("runs", $"delivery.runs.total").drop($"col").drop($"delivery").drop("overr_s").distinct().withColumn("match_id", array("created", "city", "team")).drop("created").drop("city")

// Extract relevant data from ipl_json 
val df_over6 = df_city.where($"over_id" === 5).select($"match_id", $"runs" as "runs_over6", $"team").groupBy("match_id").agg(sum("runs_over6"))
val df_over10 = df_city.where($"over_id" === 9).select($"match_id" as "match_id_10", $"runs" as "runs_over10", $"team" as "team10").groupBy("match_id_10").agg(sum("runs_over10"))
val df_over15 = df_city.where($"over_id" === 14).select($"match_id" as "match_id_15", $"runs" as "runs_over15", $"team" as "team15").groupBy("match_id_15").agg(sum("runs_over15"))
val df_over20 = df_city.where($"over_id" === 19).select($"match_id" as "match_id_20", $"runs" as "runs_over20", $"team" as "team20").groupBy("match_id_20").agg(sum("runs_over20"))

val temp_join1 = df_over6.join(df_over10, df_over6("match_id") === df_over10("match_id_10"), "inner").drop("match_id_10").drop("team10")
val temp_join2 = df_over15.join(df_over20, df_over15("match_id_15") === df_over20("match_id_20"), "inner").drop("match_id_20").drop("team15").drop("team20")
val result_join = temp_join1.join(temp_join2, temp_join1("match_id") === temp_join2("match_id_15"), "inner").drop("match_id_15")

// Fields: team, total_runs_over6, total_runs_over10, total_runs_over15, total_runs_over20
val mined_df = result_join.withColumn("team", $"match_id".getItem(2)).drop("match_id").select($"team", $"sum(runs_over6)" as "total_runs_over6", $"sum(runs_over10)" as "total_runs_over10", $"sum(runs_over15)" as "total_runs_over15", $"sum(runs_over20)" as "total_runs_over20")

// List of all the teams
val teams_list = mined_df.select($"team").distinct()

// For one team
val team_df = mined_df.where($"team" === "Mumbai Indians").select($"*")

// Apply linear regression
val df_l1 = team_df.select($"total_runs_over20".as("label"), $"total_runs_over6", $"total_runs_over10", $"total_runs_over15")
val assembler = new VectorAssembler().setInputCols(Array("total_runs_over6", "total_runs_over10", "total_runs_over15")).setOutputCol("features")
val output = assembler.transform(df_l1).select($"label",$"features")

// Split data to have 80% training data and 20% test data
val Array(training, test) = output.select("label","features").randomSplit(Array(0.8, 0.2), seed = 12345)
val lr = new LinearRegression()
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,Array(10000,0.1)).build()
val trainValidationSplit = (new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator()).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8) )
val model = trainValidationSplit.fit(training)
model.transform(test).select("features", "label", "prediction").show()
