package net.recommenders.tutorial;

import net.recommenders.rival.core.DataModel;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.core.Parser;
import net.recommenders.rival.core.SimpleParser;
import net.recommenders.rival.evaluation.metric.error.RMSE;
import net.recommenders.rival.evaluation.metric.ranking.NDCG;
import net.recommenders.rival.evaluation.metric.ranking.Precision;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.examples.DataDownloader;
import net.recommenders.rival.recommend.frameworks.RecommenderIO;
import net.recommenders.rival.recommend.frameworks.exceptions.RecommenderException;
import net.recommenders.rival.recommend.frameworks.mahout.GenericRecommenderBuilder;
import net.recommenders.rival.split.parser.MovielensParser;
import net.recommenders.rival.split.splitter.CrossValidationSplitter;
import org.apache.commons.cli.*;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

/**
 * RiVal Movielens100k Mahout Example, using 5-fold cross validation.
 *
 * @author <a href="http://github.com/alansaid">Alan</a>
 */
public final class CrossValidation {

  /**
   * Default number of folds.
   */
  public static final int N_FOLDS = 5;
  /**
   * Default neighbohood size.
   */
  public static int NEIGH_SIZE = 50;
  /**
   * Default cutoff for evaluation metrics.
   */
  public static final int AT = 10;
  /**
   * Default relevance threshold.
   */
  public static double REL_TH = 3.0;
  /**
   * Default per user setting
   */
  public static boolean PER_USER = true;
  /**
   * Default seed.
   */
  public static final long SEED = 2048L;

  /**
   * Utility classes should not have a public or default constructor.
   */
  private CrossValidation() {
  }

  /**
   * Main method. Parameter is not used.
   *
   * @param args the arguments (not used)
   */
  public static void main(final String[] args) throws FileNotFoundException, UnsupportedEncodingException, ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException, ParseException, RecommenderException {


    parseCLI(args);


    String url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip";
    String folder = "data/ml-100k";
    String modelPath = "data/ml-100k/model/";
    String recPath = "data/ml-100k/recommendations/";
    String dataFile = "data/ml-100k/u.data";

    if(!new File(dataFile).exists()) {
      DataDownloader dd = new DataDownloader(url, folder);
      dd.downloadAndUnzip();
    }
    prepareSplits(url, N_FOLDS, dataFile, folder, modelPath);
    recommend(N_FOLDS, modelPath, recPath);
    prepareStrategy(N_FOLDS, modelPath, recPath, modelPath);
    evaluate(N_FOLDS, modelPath, recPath);
  }

  /**
   * Parses the command line arguments
   * @param args the arguments
   * @throws ParseException if parsing breaks
   */
  private static void parseCLI(String[] args) throws ParseException {
    Options options = new Options();
    options.addOption("t", true, "threshold");
    options.addOption("u", true, "per user");
    options.addOption("n", true, "neighborhood size");
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = parser.parse(options, args);
    REL_TH = (null != cmd.getOptionValue("t") ? Double.parseDouble(cmd.getOptionValue("t")) : REL_TH);
    PER_USER = (null != cmd.getOptionValue("u") ? Boolean.parseBoolean(cmd.getOptionValue("u")) : PER_USER);
    NEIGH_SIZE = (null != cmd.getOptionValue("u") ? Integer.parseInt(cmd.getOptionValue("n")) : NEIGH_SIZE);
  }


  /**
   * Downloads a dataset and stores the splits generated from it.
   *
   * @param url url where dataset can be downloaded from
   * @param nFolds number of folds
   * @param inFile file to be used once the dataset has been downloaded
   * @param folder folder where dataset will be stored
   * @param outPath path where the splits will be stored
   */
  public static void prepareSplits(final String url, final int nFolds, final String inFile, final String folder, final String outPath) throws FileNotFoundException, UnsupportedEncodingException {

    boolean perUser = PER_USER;
    long seed = SEED;
    Parser parser = new MovielensParser();

    DataModel<Long, Long> data = null;
    try {
      data = parser.parseData(new File(inFile));
    } catch (IOException e) {
      e.printStackTrace();
    }

    DataModel<Long, Long>[] splits = new CrossValidationSplitter(nFolds, perUser, seed).split(data);
    File dir = new File(outPath);
    if (!dir.exists()) {
      if (!dir.mkdir()) {
        System.err.println("Directory " + dir + " could not be created");
        return;
      }
    }
    for (int i = 0; i < splits.length / 2; i++) {
      DataModel<Long, Long> training = splits[2 * i];
      DataModel<Long, Long> test = splits[2 * i + 1];
      String trainingFile = outPath + "train_" + i + ".csv";
      String testFile = outPath + "test_" + i + ".csv";
      System.out.println("train: " + trainingFile);
      System.out.println("test: " + testFile);
      boolean overwrite = true;
      DataModelUtils.saveDataModel(training, trainingFile, overwrite);
      DataModelUtils.saveDataModel(test, testFile, overwrite);
    }
  }

  /**
   * Recommends using an UB algorithm.
   *
   * @param nFolds number of folds
   * @param inPath path where training and test models have been stored
   * @param outPath path where recommendation files will be stored
   */
  public static void recommend(final int nFolds, final String inPath, final String outPath) throws RecommenderException {
    for (int i = 0; i < nFolds; i++) {
      org.apache.mahout.cf.taste.model.DataModel trainModel;
      org.apache.mahout.cf.taste.model.DataModel testModel;
      try {
        trainModel = new FileDataModel(new File(inPath + "train_" + i + ".csv"));
        testModel = new FileDataModel(new File(inPath + "test_" + i + ".csv"));
      } catch (IOException e) {
        e.printStackTrace();
        return;
      }

      GenericRecommenderBuilder grb = new GenericRecommenderBuilder();
      String recommenderClass = "org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender";
      String similarityClass = "org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity";
      int neighborhoodSize = NEIGH_SIZE;
      Recommender recommender = null;
      try {
        recommender = grb.buildRecommender(trainModel, recommenderClass, similarityClass, neighborhoodSize);
      } catch (RecommenderException e) {
        e.printStackTrace();
      }

      String fileName = "recs_" + i + ".csv";

      LongPrimitiveIterator users;
      try {
        users = testModel.getUserIDs();
        boolean createFile = true;
        while (users.hasNext()) {
          long u = users.nextLong();
          assert recommender != null;
          List<RecommendedItem> items = recommender.recommend(u, trainModel.getNumItems());
          RecommenderIO.writeData(u, items, outPath, fileName, !createFile, null);
          createFile = false;
        }
      } catch (TasteException e) {
        e.printStackTrace();
      }
    }
  }

  /**
   * Prepares the strategies to be evaluated with the recommenders already
   * generated.
   *
   * @param nFolds number of folds
   * @param splitPath path where splits have been stored
   * @param recPath path where recommendation files have been stored
   * @param outPath path where the filtered recommendations will be stored
   */
  @SuppressWarnings("unchecked")
  public static void prepareStrategy(final int nFolds, final String splitPath, final String recPath, final String outPath) throws InstantiationException, IllegalAccessException, NoSuchMethodException, ClassNotFoundException, InvocationTargetException, FileNotFoundException, UnsupportedEncodingException {
    for (int i = 0; i < nFolds; i++) {
      File trainingFile = new File(splitPath + "train_" + i + ".csv");
      File testFile = new File(splitPath + "test_" + i + ".csv");
      File recFile = new File(recPath + "recs_" + i + ".csv");
      DataModel<Long, Long> trainingModel;
      DataModel<Long, Long> testModel;
      DataModel<Long, Long> recModel;
      try {
        trainingModel = new SimpleParser().parseData(trainingFile);
        testModel = new SimpleParser().parseData(testFile);
        recModel = new SimpleParser().parseData(recFile);
      } catch (IOException e) {
        e.printStackTrace();
        return;
      }

      Double threshold = REL_TH;
      String strategyClassName = "net.recommenders.rival.evaluation.strategy.UserTest";
      EvaluationStrategy<Long, Long> strategy = null;
      strategy = (EvaluationStrategy<Long, Long>) (Class.forName(strategyClassName)).getConstructor(DataModel.class, DataModel.class, double.class).
          newInstance(trainingModel, testModel, threshold);

      DataModel<Long, Long> modelToEval = new DataModel();
      for (Long user : recModel.getUsers()) {
        assert strategy != null;
        for (Long item : strategy.getCandidateItemsToRank(user)) {
          if (recModel.getUserItemPreferences().get(user).containsKey(item)) {
            modelToEval.addPreference(user, item, recModel.getUserItemPreferences().get(user).get(item));
          }
        }
      }
      DataModelUtils.saveDataModel(modelToEval, outPath + "strategymodel_" + i + ".csv", true);
    }
  }

  /**
   * Evaluates the recommendations generated in previous steps.
   *
   * @param nFolds number of folds
   * @param splitPath path where splits have been stored
   * @param recPath path where recommendation files have been stored
   */
  public static void evaluate(final int nFolds, final String splitPath, final String recPath) {
    double ndcgRes = 0.0;
    double precisionRes = 0.0;
    double rmseRes = 0.0;
    for (int i = 0; i < nFolds; i++) {
      File testFile = new File(splitPath + "test_" + i + ".csv");
      File recFile = new File(recPath + "recs_" + i + ".csv");
      DataModel<Long, Long> testModel = null;
      DataModel<Long, Long> recModel = null;
      try {
        testModel = new SimpleParser().parseData(testFile);
        recModel = new SimpleParser().parseData(recFile);
      } catch (IOException e) {
        e.printStackTrace();
      }
      NDCG ndcg = new NDCG(recModel, testModel, new int[]{AT});
      ndcg.compute();
      ndcgRes += ndcg.getValueAt(AT);

      RMSE rmse = new RMSE(recModel, testModel);
      rmse.compute();
      rmseRes += rmse.getValue();

      Precision precision = new Precision(recModel, testModel, REL_TH, new int[]{AT});
      precision.compute();
      precisionRes += precision.getValueAt(AT);
    }
    System.out.println("NDCG@" + AT + ": " + ndcgRes / nFolds);
    System.out.println("RMSE: " + rmseRes / nFolds);
    System.out.println("P@" + AT + ": " + precisionRes / nFolds);

  }
}
