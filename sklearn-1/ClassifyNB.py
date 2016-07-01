def classify(features_train, labels_train):
	from sklearn.naive_bayes import GaussianNB ### sklearn module imported
	clf = GaussianNB ### classifier created
	clf.fit(features_train, labels_train) ### Classifier fit on training features and labels
	pred = clf.predict(features_test) ### Fit Classifier returned
