package com.oracle.ml.demo;

import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;


public class MLDemo {




    public static void main(String[]args) throws IOException {



        var regressionFactory = new RegressionFactory();
        var csvLoader = new CSVLoader<>(regressionFactory);


        //Load data
        var insuranceHeaders = new String[]{"age","sexe","imc","enfants","fumeur","departement","prime"};
        var insuranceDataSource = csvLoader.loadDataSource(Paths.get("src/main/resources/insurance.csv"),"prime",insuranceHeaders);

        //split train and test data
        var splitter = new TrainTestSplitter<>(insuranceDataSource,0.95, 0L);
        var trainingDataset = new MutableDataset<>(splitter.getTrain());
        var  testingDataset = new MutableDataset<>(splitter.getTest());

        //display data overwiew
        System.out.println(String.format("Training data size = %d, number of features = %d",trainingDataset.size(),trainingDataset.getFeatureMap().size()));
        System.out.println(String.format("Testing data size = %d, number of features = %d",testingDataset.size(),testingDataset.getFeatureMap().size()));


        // create and Train Model on Train data set
        var xgb = new XGBoostRegressionTrainer(50);
        var xgbModel = xgb.train(trainingDataset);


        // Make prediction
        Example<Regressor> xPredict = testingDataset.getExample(0);
        System.out.println(xPredict);
        List<Prediction<Regressor>> yPredict = Collections.singletonList(xgbModel.predict(xPredict));
        System.out.println(yPredict);

        // Evaluate Model towards TestingDataSet
        RegressionEvaluator evaluator = new RegressionEvaluator();
        var evaluation = evaluator.evaluate(xgbModel,trainingDataset);


        System.out.println(evaluation.toString());



    }



}
