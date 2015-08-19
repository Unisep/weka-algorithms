/*
 * The MIT License
 *
 * Copyright 2015 brunoocasali.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package wekaalgorithms;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author brunoocasali
 */
public class WekaAlgorithms {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("src/files/letter.arff");

        int folds = 10;
        int runs = 30;

        Classifier cls = new NaiveBayes();
        Instances data = source.getDataSet();
        data.setClassIndex(16);

        System.out.println("#seed \t correctly instances \t percentage of corrects\n");
        for (int i = 1; i <= runs; i++) {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, folds, new Random(i));

            System.out.println("#" + i + "\t" + summary(eval));
        }
    }
    
    private static String summary(Evaluation eval){
        return Utils.doubleToString(eval.correct(), 12, 4) + "\t " +
                Utils.doubleToString(eval.pctCorrect(), 12, 4) + "%";
    }
}
