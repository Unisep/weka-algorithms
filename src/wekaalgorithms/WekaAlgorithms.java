/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaalgorithms;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author bruno
 */
public class WekaAlgorithms {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        DataSource data = new DataSource("src/letter.arff");
        Instances ins = data.getDataSet();

        NaiveBayes naive = new NaiveBayes();
        ins.setClassIndex(16);
        naive.buildClassifier(ins);

        Evaluation eval = new Evaluation(ins);

        printSummary(naive, eval, ins);
//        System.out.println(ins.toString());
//        
//        
//        System.out.println(naive.toString());
//        
//        Instance k = new Instance(5);
//        k.setDataset(ins);
//        k.setValue(0, "sunny");
//        k.setValue(1, "mild");
//        k.setValue(2, "normal");
//        k.setValue(3, "FALSE");
//        
//        double prob[] = naive.distributionForInstance(k);
//        System.out.println("YES: " + prob[0]);
//        System.out.println("NO: " + prob[1]);

        // loads data and set class index
    }

    private static void printSummary(Classifier base, Evaluation eval, Instances data) throws Exception {
        // output evaluation
        System.out.println();
        System.out.println("=== Setup ===");
//        System.out.println("Classifier: " + classifierName.getClass().getName() + " " + Utils.joinOptions(base.getOptions()));
        System.out.println("Dataset: " + data.relationName());
        System.out.println();

        // output predictions
        System.out.println("# - actual - predicted - error - distribution - token");
        for (int i = 0; i < data.numInstances(); i++) {
            double pred = base.classifyInstance(data.instance(i));
            double actual = data.instance(i).classValue();
            double[] dist = base.distributionForInstance(data.instance(i));

            if (pred != actual) {
                System.out.print((i + 1));
                System.out.print(" - ");
                System.out.print(data.instance(i).toString(data.classIndex()));
                System.out.print(" - ");
                System.out.print(data.classAttribute().value((int) pred));
                System.out.print(" - ");
                if (pred != data.instance(i).classValue()) {
                    System.out.print("yes");
                } else {
                    System.out.print("no");
                }
                System.out.print(" - ");
                System.out.print(Utils.arrayToString(dist));
                System.out.print(" - ");
                data.instance(i).enumerateAttributes().toString();
                System.out.println();
            }
        }

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
