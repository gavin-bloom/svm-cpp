#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

template <typename T>
class SVM {
private:
    vector<vector<T>> X; // feature vectors
    vector<T> y; // labels
    vector<T> alpha; // Lagrange multipliers
    T b; // bias
    T C; // regularization parameter
    T tol; // numerical tolerance
    int max_iter; // maximum number of iterations
    T sigma; // parameter for Gaussian kernel
    
    //Gaussian kernel
    T kernel(const vector<T>& x1, const vector<T>& x2) const {
        T dist_sq = 0;
        for (int i = 0; i < x1.size(); ++i) {
            dist_sq += pow(x1[i] - x2[i], 2);
        }
        return exp(-dist_sq / (2 * sigma * sigma));
    }
    
public:
    void fit(const vector<vector<T>>& X_, const vector<T>& y_, T C_, T tol_, int max_iter_) {
        X = X_;
        y = y_;
        alpha.resize(X.size()); 
        b = 0;
        tol = tol_;
        C = C_;
        max_iter = max_iter_;
        
	//SMO loop
        int iter = 0;
        while (iter < max_iter) {
            int num_changed_alphas = 0;
            for (int i = 0; i < X.size(); ++i) {
                T f_i = b;
                for (int j = 0; j < X.size(); ++j) {
                    f_i += alpha[j] * y[j] * kernel(X[j], X[i]); //predicted output for sample i
                }
                T E_i = f_i - y[i]; //error for sample i
		//check if error violates Karush–Kuhn–Tucker conditions
                if ((y[i] * E_i < -tol && alpha[i] < C) || (y[i] * E_i > tol && alpha[i] > 0)) {
                    //select random sample j and compute error
		    int j = i;
                    while (j == i) {
                        j = rand() % X.size();
		    }
                    T f_j = b;
                    for (int k = 0; k < X.size(); ++k) {
                        f_j += alpha[k] * y[k] * kernel(X[k], X[j]);
                    }
                    T E_j = f_j - y[j];
			
                    T alpha_i_old = alpha[i];
                    T alpha_j_old = alpha[j];
			
                    T L, H; //compute low and high bounds for alpha coefficients
                    if (y[i] != y[j]) {
                        L = max(0.0, alpha[j] - alpha[i]);
                        H = min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = max(0.0, alpha[i] + alpha[j] - C);
                        H = min(C, alpha[i] + alpha[j]);
                    }
                    if (L == H) {
                        continue;
                    }
		    
	            //compute eta and make sure that the second derivative of the objective function with respect to alpha[j] is negative
		    //no maximum if not
                    T eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j]);
                    if (eta >= 0) {
                        continue;
                    }
		    
	            //comput new alpha for sample j and check tolerance
                    alpha[j] -= y[j] * (E_i - E_j) / eta;
                    alpha[j] = min(max(alpha[j], L), H);
                    if (abs(alpha[j] - alpha_j_old) < tol) {
                        alpha[j] = alpha_j_old;
                        continue;
                    }
			
		    //compute new alpha for sample i
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j]);
		    //compute bias
                    T b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[i]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[i], X[j]);
					T b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[j]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[j], X[j]);
                    if (alpha[i] > 0 && alpha[i] < C) {
                        b = b1;
                    } else if (alpha[j] > 0 && alpha[j] < C) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2;
                    }
		   
                    ++num_changed_alphas;
                }
            }
            
            if (num_changed_alphas == 0) {
                ++iter;
            } else {
                iter = 0;
            }
        }
    }
    
    T predict(const vector<T>& x) const {
        T f = b;
        for (int i = 0; i < X.size(); ++i) {
            f += alpha[i] * y[i] * kernel(X[i], x);
        }
        return (f >= 0) ? 1 : -1;
    }
};

// Sample usage
int main() {
    SVM<double> svm;
    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> y = {-1, 1, 1, -1};
    svm.fit(X, y, 1.0, 0.0001, 10000);
	vector<vector<double>> X_test = {{0.5, 0.5}, {0.3, 0.7}, {0.8, 0.8}, {0.3, 0.0} };
	for (const auto& x : X_test) {
		cout << "Prediction for (" << x[0] << ", " << x[1] << "): " << svm.predict(x) << endl;
	}
    return 0;
}
