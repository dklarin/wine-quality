from flask import Blueprint, render_template


reg_lin_page = Blueprint('reg_lin_page', __name__,
                         template_folder='templates')

sp = ['reg_lin_page.', 'reg_lin_gra_spu_page.', 'reg_lin_una_val_page.']

reg_lin_links = ['gradijentni_spust', 'unakrsna_validacija']
gra_spu_links = ['distribution_alcohol', 'distribution_quality',
                 'gradient_descen', 'funkcija_troska', 'reg_lin_izgled_regresije']
una_val_links = ['reg_lin_metrike']


# 1
@reg_lin_page.route('/regression_linear')
def regression_linear():

    return render_template(
        'info.html',
        link1=sp[0]+reg_lin_links[0],
        link2=sp[0]+reg_lin_links[1],
        title='linear regression')


# 1.1
@reg_lin_page.route('/gradijentni_spust')
def gradijentni_spust():

    return render_template(
        'info.html',
        link1=sp[1]+gra_spu_links[0],
        link2=sp[1]+gra_spu_links[1],
        link3=sp[1]+gra_spu_links[2],
        link4=sp[1]+gra_spu_links[3],
        link5=sp[1]+gra_spu_links[4],
        title='gradijentni spust'
    )


# 1.2
@reg_lin_page.route('/unakrsna_validacija')
def unakrsna_validacija():

    return render_template(
        'info.html',
        link1=sp[2]+una_val_links[0],
        link2=sp[2]+una_val_links[0],
        link3=sp[2]+una_val_links[0],
        title='unakrsna validacija'
    )
