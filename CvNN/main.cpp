/*=============================================================================
 * Project : CvNN
 * Code : main.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/11/21
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Test Neurarl Network
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

/*=== Local Define / Local Const ============================================*/

static cv::Range inputrange; /**< 入力データ範囲の指定 */
static cv::Range targetrange; /**< 正解データ範囲の指定 */

/*=== Local Variable ========================================================*/
/** XOR問題.
 * 末列 : 目標値 その他 : 入力
 */
static float_t testdata[4][3] = {
    {0.9, 0.9, 0.1},
    {0.9, 0.1, 0.9},
    {0.1, 0.9, 0.9},
    {0.9, 0.9, 0.1}
};


/*=== Local Function Define =================================================*/

/** 学習データを入力部分と目標値部分に分割する.
 * @param traindate 学習データ
 * @param inputs NN入力データ
 * @param target 目標値データ
 * @return true / 成功, false / 失敗
 */
static bool separateInputAndTarget(cv::Mat& traindata, cv::Mat& inputs, cv::Mat& target);

/** NNをテストし,保存する.
 * @param model NNモデル
 * @param inputs NN入力データ
 * @param target 目標値
 * @param numtrainsamples サンプルデータの数
 * @param filename NNモデル保存ファイル名
 * @return なし
 */
static void testAndSaveNN(const cv::Ptr<cv::ml::StatModel>& model,
                          const cv::Mat& inputs, const cv::Mat& target,
                          int numtrainsamples,
                          const std::string& filename);

/** NNを構築する.
 * この関数は内部でseparateInputAndTargetと
 * testAndSaveNNを呼び出す.
 * @param traindate 学習データ
 * @param savefilename NNモデル保存ファイル名
 * @return true / 成功, false / 失敗
 */
static bool buildNN(cv::Mat& traindata, std::string savefilename);

/** NNをロードし,テストする.
 * この関数は内部でseparateInputAndTargetと
 * testAndSaveNNを呼び出す.
 * @param traindate 学習データ
 * @param inputfilename NNモデルファイル名
 * @return true / 成功, false / 失敗
 */
static bool loabAndTestNN(cv::Mat& traindata, std::string inputfilename);

/*=== Local Function Implementation =========================================*/
bool separateInputAndTarget(cv::Mat &traindata, cv::Mat& inputs, cv::Mat& target)
{
    /* 引数チェック */
    if(traindata.empty()){
      LOG(ERROR) << "traindata is invalid" << std::endl;
      return false;
    }

    /* 先頭から末列-1を格納 */
    //cv::Mat temp = traindata.colRange(0, traindata.cols - 1);
    cv::Mat temp = traindata.colRange(inputrange);
    inputs = temp.clone();

    /* 末列を格納 */
    //temp = traindata.col(traindata.cols - 1);
    temp = traindata.colRange(targetrange);
    target = temp.clone(); // 参照のまま渡すとNN作成時にエラーになる
                           // 行列が連続であることが内部でチェックされている
      return true;
}

void testAndSaveNN(const cv::Ptr<cv::ml::StatModel>& model,
                   const cv::Mat& inputs, const cv::Mat& target,
                   int numsamples,
                   const std::string& filename)
{
    int i, samplesnum = inputs.rows;

    /* 入力値,出力値,確度を出力 */
    for(i = 0; i < samplesnum; i++){
        cv::Mat sample = inputs.row(i);
        cv::Mat result;
        /* resultには予想結果が格納される */
        float_t err_rate = model->predict(sample, result);

        LOG(INFO) << "Input : " << sample.rowRange(cv::Range::all())
                  << "Output : " << result.rowRange(cv::Range::all())
                  << "Rate : " << std::to_string(err_rate) << std::endl;

    }
    /* 保存 */
    if(!filename.empty()){
        model->save(filename);
    }
}

bool buildNN(cv::Mat& traindata, std::string savefilename)
{
    cv::Mat inputs;
    cv::Mat target;
    /* データの分割 */
    separateInputAndTarget(traindata, inputs, target);
    LOG(INFO) << traindata.rowRange(cv::Range::all()) << std::endl;
    LOG(INFO) << inputs.rowRange(cv::Range::all()) << std::endl;
    LOG(INFO) << target.rowRange(cv::Range::all()) << std::endl;

    cv::Ptr<cv::ml::ANN_MLP> nnmodel;

    /* NN設定 */
    /* 入力層,中間層,出力層を指定
     * 先頭:入力層, 末尾:出力層, その他:中間層
     * 各配列の数値がユニット数になる
     */
    int layerlayout[] = {inputs.cols, target.cols}; // 中間層は2ユニット以上必要
    cv::Mat layer(1, 2, CV_32S, layerlayout);
    /* アルゴリズムの指定 */
    int method = cv::ml::ANN_MLP::Params::BACKPROP;
    /* アルゴリズムに与えるパラメータ
     * ここでは誤差逆伝搬の修正量
     */
    double methodparam0 = 0.1;
    /* アルゴリズムに与えるパラメータ
     * ここでは誤差逆伝搬の慣性項
     */
    double methodparam1 = 0.0;
    /* 学習のループ数 */
    int maxiterration = 10000;
    /* ループのパラメータ */
    cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::COUNT, maxiterration, 0);

    LOG(INFO) << layer.rowRange(cv::Range::all()) << std::endl;
    cv::ml::ANN_MLP::Params params = cv::ml::ANN_MLP::Params(layer,
                                                             cv::ml::ANN_MLP::SIGMOID_SYM,
                                                             0, 0,
                                                             termcriteria,
                                                             method, methodparam0, methodparam1);

    /* 学習データセット作成 */
    cv::Ptr<cv::ml::TrainData> train = cv::ml::TrainData::create(inputs, cv::ml::ROW_SAMPLE, target);
    cv::Mat traininput = train->getTrainSamples(); // デバッグ用
    cv::Mat trainoutput = train->getTrainResponses(); // デバッグ用

    /* NN構築 */
    nnmodel = cv::ml::StatModel::train<cv::ml::ANN_MLP>(train, params);

    /* 学習結果の確認と保存 */
    testAndSaveNN(nnmodel, inputs, target, inputs.rows, savefilename);

    return true;
}

bool loabAndTestNN(cv::Mat& traindata, std::string inputfilename)
{
    cv::Mat inputs;
    cv::Mat target;
    /* データの分割 */
    separateInputAndTarget(traindata, inputs, target);
    LOG(INFO) << traindata.rowRange(cv::Range::all()) << std::endl;
    LOG(INFO) << inputs.rowRange(cv::Range::all()) << std::endl;
    LOG(INFO) << target.rowRange(cv::Range::all()) << std::endl;

    cv::Ptr<cv::ml::ANN_MLP> nnmodel;

    if(!inputfilename.empty()){
        nnmodel = cv::ml::StatModel::load<cv::ml::ANN_MLP>(inputfilename);
        if(nnmodel.empty()){
            LOG(ERROR) << "Can not read model!" << std::endl;
            return false;
        }else{
            LOG(INFO) << "Load " << inputfilename << "success!!" << std::endl;
        }
    }
    /* テスト */
    testAndSaveNN(nnmodel, inputs, target, inputs.rows, "");
    return true;
}

/*=== Global Function Implementation ========================================*/

int main(int argc, char *argv[]) {
    /* Initialize */
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    if(argc < 3){
        LOG(INFO) << "Usage: CvNN [mode 0:train and save, 1:load and test] [modelname(when mode=0 savefilename, mode=1 inputfilename)]" << std::endl;
        return EXIT_FAILURE;
    }

    /* 学習データの作成 */
    cv::Mat traindata = cv::Mat::zeros(4, 3, CV_32F);
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 3; j++){
            traindata.at<float>(i, j) = testdata[i][j];
        }
    /* 学習データの内,どこから入力でどこから目標値かを指定 */
    inputrange = cv::Range(0, 2);
    targetrange = cv::Range(2, 3);

    /* 学習 & 保存 */
    if(0 == std::atoi(argv[1])){
        std::string savefilename = argv[2];
        if(savefilename.empty())
            savefilename = "out.data";
        buildNN(traindata, savefilename);
    } /* ロード & test */
    else if(1 == std::atoi(argv[1])){
        std::string inputfilename = argv[2];
        if(inputfilename.empty())
            inputfilename = "out.data";
        loabAndTestNN(traindata, inputfilename);
    }

    /* Finalize */
    google::InstallFailureSignalHandler();

    return EXIT_SUCCESS;
}
