from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

reg_lin = Blueprint('regression_linear_page', __name__,
                    template_folder='templates')


@reg_lin.route('/linear_regression')
def linear_regression():

    return render_template(
        'info.html',
        link1='gradijentni_spust', link2='unakrsna_validacija', title='linear regression')
