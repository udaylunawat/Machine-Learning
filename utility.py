def roc_auc_pipe(clf, X_train, X_test, y_train, y_test, title):
    from sklearn.metrics import roc_curve, auc
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]

    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)

    plt.close
    plt.plot(train_fpr, train_tpr, label="train AUC = {:.2f}".format(train_auc))
    plt.plot(test_fpr, test_tpr, label="test AUC = {:.2f}".format(test_auc))
    plt.legend()
    plt.plot([0, 1], [0, 1],'g--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate(FPR)")
    plt.ylabel("True Positive Rate(TPR)")
    plt.title(title)
    plt.grid()
    plt.show()

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


#When grid search using single hyperparameters

def singleplot(rs, alpha, title):
  '''Helps in plotting single parameter GridSearch/RandomSearch plots'''

  print(("best from search: %.3f"
        % rs.score(X_test, y_test)))

  results = pd.DataFrame.from_dict(rs.cv_results_)

  results = results.sort_values(alpha)
  train_auc= results['mean_train_score']
  train_auc_std= results['std_train_score']
  cv_auc = results['mean_test_score'] 
  cv_auc_std= results['std_test_score']

  K =  results[alpha]

  plt.plot(K, train_auc, label='Train AUC')
  # https://stackoverflow.com/a/48803361/4084039
  # plt.gca().fill_between(K, train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

  plt.plot(K, cv_auc, label='CV AUC')

  plt.scatter(K, train_auc, label='Train AUC points')
  plt.scatter(K, cv_auc, label='CV AUC points')


  plt.legend()
  plt.xscale('log')
  plt.xlabel("Alpha")
  plt.ylabel("AUC")
  plt.title(title)
  plt.grid()
  plt.show()

  results.head()


def colab_gcp():
  #connect Colab to GCS using Google Auth API and gsutil
  from google.colab import auth
  auth.authenticate_user()
  project_id = 'appliedai-2020'
  !gcloud config set project {project_id}
  !gsutil ls

  # Mounting GCS data bucket
  !echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
  !curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
  !apt -qq update  &> /dev/null
  !apt -qq install gcsfuse  &> /dev/null

  bucket_name = 'dracarys3_bucket'
  !mkdir bucket
  !gcsfuse $bucket_name /content/bucket


# python download function
import requests
def download_file(url, file_name):
  r = requests.get(url, allow_redirects=True)
  open(file_name, 'wb').write(r.content)

# python unzip function
import zipfile
def unzip(file_name, directory):
  with zipfile.ZipFile(file_name, 'r') as zip_ref:
      zip_ref.extractall(directory)