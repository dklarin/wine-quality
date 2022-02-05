from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

reg_pol_page = Blueprint('reg_pol_page', __name__,
                         template_folder='templates')


@reg_pol_page.route('/', defaults={'page': 'index'})
@reg_pol_page.route('/<page>')
def show(page):
    try:
        return render_template(f'pages/{page}.html')
    except TemplateNotFound:
        abort(404)


@reg_pol_page.route('/polynomial_regression')
def polynomial_regression():

    return render_template(
        'info.html',
        link1='index', link2='index', title='polynomial regression')
