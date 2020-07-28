import pickle
import urllib


def load_idx_to_label(dataset_name):
    """Return a dictionary containing the mapping from the index
    of a label to the actual label name.
    """
    if dataset_name == 'imagenet':
        path = 'https://gist.githubusercontent.com/yrevar/'
        path += '6135f1bd8dcf2e0cc683/raw/'
        path += 'd133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee'
        path += '/imagenet1000_clsid_to_human.pkl'
        idx_to_label = pickle.load(urllib.request.urlopen(path))
        
    elif dataset_name == 'indoor_scenes':
        label_to_idx = {'airport_inside': 0,
                        'bar': 1,
                        'bedroom': 2,
                        'casino': 3,
                        'inside_subway': 4,
                        'kitchen': 5,
                        'livingroom': 6,
                        'restaurant': 7,
                        'subway': 8,
                        'warehouse': 9}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
    elif dataset_name == 'pubfig10':
        celebs = ['Aaron-Eckhart', 'Adriana-Lima',
                  'Angela-Merkel', 'Beyonce-Knowles', 
                  'Brad-Pitt', 'Clive-Owen', 
                  'Drew-Barrymore', 'Milla-Jovovich', 
                  'Quincy-Jones', 'Shahrukh-Khan']
        idx_to_label = { i: celebs[i] for i in range(len(celebs)) }

    elif dataset_name == 'pubfig83':
        celebs = ['adam-sandler', 'alex-baldwin', 'angelina-jolie', 'anna-kournikova', 'ashton-kutcher', 'avril-lavigne',
                 'barack-obama', 'ben-affleck', 'beyonce-knowles', 'brad-pitt', 'cameron-diaz', 'cate-blanchett', 'charlize-theron',
                 'christina-ricci', 'claudia-schiffer', 'clive-owen', 'colin-farell', 'colin-powell', 'cristiano-ronaldo', 'daniel-craig',
                 'daniel-radcliffe', 'david-beckham', 'david-duchovny', 'denise-richards', 'drew-barrymore', 'dustin-hoffman', 'ehud-olmert',
                 'eva-mendes', 'faith-hill', 'george-clooney', 'gordon-brown', 'gwyneth-paltrow', 'halle-berry', 'harrison-ford',
                 'hugh-jackman', 'hugh-laurie', 'jack-nicholson', 'jennifer-aniston', 'jennifer-lopez', 'jennifer-lovehewitt',
                 'jessica-alba', 'jessica-simpson', 'joaquin-phoenix', 'john-travolta', 'julia-roberts', 'jula-stiles', 'kate-moss',
                 'kate-winslet', 'katherine-heigl', 'keira-knightley', 'kiefer-sutherland', 'leonardo-dicaprio', 'lindsay-lohan', 'mariah-carey',
                 'martha-stewart', 'matt-damon', 'meg-ryan', 'meryl-streep', 'michael-bloomberg', 'mickey-rourke', 'miley-cyrus',
                 'morgan-freeman', 'nicole-kidman', 'nicole-richie', 'orlando-bloom', 'reese-witherspoon', 'renee-zellweger', 'ricky-martin',
                 'robert-gates', 'sania-mirza', 'scarlett-johansson', 'shahrukh-khan', 'shakira', 'sharon-stone', 'silvio-berlusconi',
                 'stephen-colbert', 'steve-carell', 'tom-cruise', 'uma-thurman', 'victoria-beckham', 'viggo-mortensen', 'will-smith', 'zac-efron']
        idx_to_label = { i: celebs[i] for i in range(len(celebs)) }

    elif dataset_name == 'vggface2':
        path = "../utils/vggface2_80_to_complete.pkl"
        with open(path, 'rb') as file:
            idx_to_label = pickle.load(file)

    else:
        raise NotImplementedError
        
    return idx_to_label
