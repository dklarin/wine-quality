from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

reg_poly = Blueprint('regression_polynomial_page', __name__,
                     template_folder='templates')


@reg_poly.route('/', defaults={'page': 'index'})
@reg_poly.route('/<page>')
def show(page):
    try:
        return render_template(f'pages/{page}.html')
    except TemplateNotFound:
        abort(404)


@reg_poly.route('/polynomial_regression')
def polynomial_regression():

    return render_template(
        'info.html',
        link1='index', link2='index', title='polynomial regression')
