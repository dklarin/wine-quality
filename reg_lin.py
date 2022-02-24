from flask import Blueprint, render_template


reg_lin_page = Blueprint('reg_lin_page', __name__,
                         template_folder='templates')

sp = ['reg_lin_page.', 'reg_lin_gra_spu_page.', 'reg_lin_una_val_page.']

reg_lin_links = ['gradient_descent', 'cross_validation']
gra_spu_links = ['rlgd_distribution_alcohol', 'rlgd_distribution_ph',
                 'rlgd_gradient_descent', 'rlgd_cost_function', 'rlgd_regression_appearance']
una_val_links = ['rlcv_metrics', 'rlcv_k_fold_validation',
                 'rlcv_matrics', 'rlcv_model_training']


# 1
@reg_lin_page.route('/regression_linear')
def regression_linear():

    return render_template(
        'info.html',
        link1=sp[0]+reg_lin_links[0],
        link2=sp[0]+reg_lin_links[1],
        title='linear regression')


# 1.1
@reg_lin_page.route('/gradient_descent')
def gradient_descent():

    return render_template(
        'info.html',
        link1=sp[1]+gra_spu_links[0],
        link2=sp[1]+gra_spu_links[1],
        link3=sp[1]+gra_spu_links[2],
        link4=sp[1]+gra_spu_links[3],
        link5=sp[1]+gra_spu_links[4],
        title='gradient descent'
    )


# 1.2
@reg_lin_page.route('/cross_validation')
def cross_validation():

    return render_template(
        'info.html',
        link1=sp[2]+una_val_links[0],
        link2=sp[2]+una_val_links[1],
        link3=sp[2]+una_val_links[2],
        link4=sp[2]+una_val_links[3],
        title='unakrsna validacija'
    )
