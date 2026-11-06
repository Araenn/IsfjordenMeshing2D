#!/usr/bin/env python3
"""Source mobile SPECFEM-Waves : un script de preparation, une configuration generee.

Sur Olivia : placer ce script dans ~/specfem-waves-mobile/config,
a cote de la copie du config.json valide, puis :
  python3 source_mobile.py
Recompiler ensuite SPECFEM-Waves avec ses modules habituels et utiliser
config_mobile.json dans le job Slurm, a la place de config.json.

Le premier test dure 0.4 s. Pour les 5 s, mettre nstep=1000000 dans
config_mobile.json (si dt=5e-6). La source adapte automatiquement la longueur
du trajet et sa rampe finale a nstep*dt. Il n'y a plus de sources.npy.

Dans config_mobile.json : speed_knots=0 donne le dipole fixe de comparaison ;
speed_knots=15.8 donne le mouvement. spacing=0.5 permet une verification de
convergence spatiale. La separation de 1 m et le dipole horizontal sont les
hypotheses de ce premier essai, pas des mesures de l'helice.

Le mouvement est approche par transfert continu entre dipoles fixes proches.
Le sinus 50 Hz et les poids sont calcules aux vrais temps des sous-etapes.
L'amplitude conserve la valeur numerique 1e7, sans calibration en puissance
de bateau. Elle n'a pas la meme energie que l'ancien Ricker de meme amplitude.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

# Seul chemin a adapter si le depot sur Olivia est installe ailleurs.
SPECFEM_DIR = Path('/cluster/home/leay/specfem-waves-mobile')
CONFIG_REFERENCE = Path(__file__).resolve().with_name('config.json')
DUREE_TEST = 0.4
PROFONDEUR_SOURCE = 7.0
VITESSE_NOEUDS = 15.8

HEADER = r"""#ifndef SPECFEM_MOVING_DIPOLE_2D_HPP
#define SPECFEM_MOVING_DIPOLE_2D_HPP

#ifndef SPECFEM_MOVING_DIPOLE_MATH_TEST
#include "Multisource.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#endif

// The same clock drives the carrier and all spatial interpolation weights.
struct MovingDipoleParameters {
    double x0{}, z0{}, speed{}, duration{}, frequency{}, amplitude{};
    double spacing{1.0}, separation{1.0}, angle_deg{}, ramp{0.1};

    KOKKOS_INLINE_FUNCTION double center(const size_t original_index) const {
        return x0 + (speed < 0 ? -1.0 : 1.0) * (original_index / 2) * spacing;
    }
    KOKKOS_INLINE_FUNCTION double weight(const double t, const double anchor) const {
        if (t < 0 || t >= duration) return 0.0;
        const double w = 1.0 - Kokkos::abs((x0 + speed*t - anchor)/spacing);
        return w > 0 ? w : 0.0;
    }
    KOKKOS_INLINE_FUNCTION double waveform(const double t) const {
        if (t < 0 || t >= duration) return 0.0;
        const double pi = 3.1415926535897932384626433832795;
        const double r = ramp < duration/2 ? ramp : duration/2;
        double envelope = 1.0;
        if (t < r) envelope = 0.5*(1.0-Kokkos::cos(pi*t/r));
        if (t > duration-r) envelope = 0.5*(1.0-Kokkos::cos(pi*(duration-t)/r));
        return envelope*Kokkos::sin(2.0*pi*frequency*t);
    }
};

#ifndef SPECFEM_MOVING_DIPOLE_MATH_TEST
inline std::vector<vec2d> moving_dipole_positions(const MovingDipoleParameters& p) {
    for (double v : {p.x0,p.z0,p.speed,p.duration,p.frequency,p.amplitude,
                     p.spacing,p.separation,p.angle_deg,p.ramp})
        if (!std::isfinite(v)) throw std::runtime_error("Moving dipole: nonfinite parameter");
    if (p.duration<=0 || p.frequency<=0 || p.spacing<=0 || p.separation<=0 || p.ramp<=0)
        throw std::runtime_error("Moving dipole: duration/frequency/spacing/separation/ramp must be positive");
    const double n = std::ceil(std::abs(p.speed)*p.duration/p.spacing);
    if (n > 1000000) throw std::runtime_error("Moving dipole: excessive number of anchors");
    const size_t count = static_cast<size_t>(n)+1;
    const double angle = p.angle_deg*3.14159265358979323846/180.0;
    const double dx = 0.5*p.separation*std::cos(angle);
    const double dz = 0.5*p.separation*std::sin(angle);
    if (p.z0+std::abs(dz)>=0)
        throw std::runtime_error("Moving dipole: both monopoles must lie below z=0");
    std::vector<vec2d> positions;
    positions.reserve(2*count);
    for (size_t j=0; j<count; ++j) {
        const double x = p.center(2*j);
        positions.emplace_back(x+dx,p.z0+dz);
        positions.emplace_back(x-dx,p.z0-dz);
    }
    return positions;
}

// Fixed SEM injection operators are computed once by MultiSourceBase2D.
// A continuous transfer between neighbouring dipoles approximates their motion.
// No full source x time x RK-stage STF array is allocated.
template<Phy2x1 PHY>
class MovingDipole2D : public MultiSourceBase2D<PHY> {
    MovingDipoleParameters parameters;
    view_c_type2d time;
    view_type accel;
    scatter_type s_accel;
    view_double centers, signed_amplitude;
    view_type_1d2gll rho;
public:
    MovingDipole2D(const MovingDipoleParameters& p, const PHY& physics,
                   const view_c_type2d& times) :
        MultiSourceBase2D<PHY>(moving_dipole_positions(p),physics),
        parameters(p),time(times),accel(physics.get_accel()),s_accel(physics.get_s_accel()),
        centers("moving_dipole_centers",this->num_in_rank),
        signed_amplitude("moving_dipole_amplitudes",this->num_in_rank),
        rho("moving_dipole_rho",this->num_in_rank)
    {
        auto hc=Kokkos::create_mirror_view(centers);
        auto ha=Kokkos::create_mirror_view(signed_amplitude);
        size_t row=0;
        for (const auto& [ispec,gll_index] : this->in_rank_gll) {
            // Use the original index associated with this element-sorted row.
            const size_t original_index=gll_index.second;
            hc(row)=p.center(original_index);
            ha(row)=original_index%2 ? -p.amplitude : p.amplitude;
            Kokkos::deep_copy(EXE_SPACE{},
                Kokkos::subview(rho,row,Kokkos::ALL,Kokkos::ALL),
                Kokkos::subview(physics.get_rho(),ispec,Kokkos::ALL,Kokkos::ALL));
            ++row;
        }
        Kokkos::deep_copy(EXE_SPACE{},centers,hc);
        Kokkos::deep_copy(EXE_SPACE{},signed_amplitude,ha);
    }

    void callback(const size_t step,const size_t stage) const override {
        if (!this->num_in_rank) return;
        const_cast<scatter_type&>(s_accel).reset_except(accel);
        const auto p=parameters;
        const auto time_=time;
        const auto centers_=centers;
        const auto amplitude_=signed_amplitude;
        const auto rho_=rho;
        const auto hx=this->h_x;
        const auto hz=this->h_z;
        const auto indices=this->ibool;
        const auto scatter=s_accel;
        Kokkos::parallel_for("add_moving_dipole",
            Kokkos::MDRangePolicy<Kokkos::Rank<DIM_3D,Kokkos::Iterate::Right,Kokkos::Iterate::Right>>(
                {0,0,0},{int(this->num_in_rank),NGLL,NGLL}),
            KOKKOS_LAMBDA(const size_t row,const size_t i,const size_t k) {
                const double t=time_(step,stage);
                const double w=p.weight(t,centers_(row));
                if (w==0) return;
                const double forcing=amplitude_(row)*w*p.waveform(t)*hx(row,i)*hz(row,k);
                const auto writer=scatter.access();
                writer(indices(row,i,k))-=forcing/rho_(row,i,k);
            });
        Kokkos::Experimental::contribute(const_cast<view_type&>(accel),s_accel);
    }
};

template<Phy2x1 PHY>
std::function<void(size_t,size_t)> make_moving_dipole_callback(
    const nlohmann::json& source,const PHY& physics,const view_c_type2d& time)
{
    if constexpr (PhyAxi<PHY>) {
        throw std::runtime_error("Moving dipole: Cartesian acoustic_2d only");
    } else {
        if (source.contains("path") || source.contains("delay"))
            throw std::runtime_error("Moving dipole: external path/delay not supported");
        const auto pos=source.at("position").get<std::vector<double>>();
        if (pos.size()!=2 || time.extent(0)<2 || time.extent(1)==0)
            throw std::runtime_error("Moving dipole: expected 2D position and at least two time steps");
        CUSTOM_REAL t0,t1;
        Kokkos::deep_copy(t0,Kokkos::subview(time,0,0));
        Kokkos::deep_copy(t1,Kokkos::subview(time,1,0));
        const double dt=static_cast<double>(t1)-static_cast<double>(t0);
        MovingDipoleParameters p;
        p.x0=pos[0]; p.z0=pos[1];
        p.duration=dt*time.extent(0);
        p.speed=source.value("speed_knots",15.8)*1852.0/3600.0;
        p.frequency=source.at("frequency").get<double>();
        p.amplitude=source.value("amplitude",1.0);
        p.spacing=source.value("spacing",1.0);
        p.separation=source.value("dipole_separation",1.0);
        p.angle_deg=source.value("dipole_angle_deg",0.0);
        p.ramp=source.value("ramp_seconds",0.1);
        const auto ptr=std::make_shared<MovingDipole2D<PHY>>(p,physics,time);
        if (MPI_Interface::is_main())
            std::cout << "Moving dipole: duration=" << p.duration << " s, speed=" << p.speed
                      << " m/s, x=" << p.x0 << " -> " << p.x0+p.speed*p.duration
                      << ", z=" << p.z0 << ", spacing=" << p.spacing << " m" << std::endl;
        if (!ptr->is_in_rank()) return {};
        return [ptr](size_t step,size_t stage){ptr->callback(step,stage);};
    }
}
#endif
#endif
"""
INCLUDE = '#include "../callback_impl/MovingDipole2D.hpp"'
MARKER = '// BEGIN LEA MOVING DIPOLE'
BRANCH = '''    // BEGIN LEA MOVING DIPOLE
    if (parameters.value("type", std::string{}) == "moving_dipole")
        return make_moving_dipole_callback(parameters, dynamic_cast<const PHY&>(physic), time);
    // END LEA MOVING DIPOLE
'''


def atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(text)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(repo, reference, duration=DUREE_TEST):
    repo = Path(repo).expanduser().resolve()
    reference = Path(reference).resolve()
    original_repo = Path('/cluster/home/leay/specfem-waves').resolve()
    if repo == original_repo:
        raise ValueError('Le depot original est protege. Utiliser specfem-waves-mobile.')
    base = json.loads(reference.read_text())
    dt = float(base['dt'])
    if not math.isfinite(dt) or not math.isfinite(duration) or min(dt,duration)<=0:
        raise ValueError('dt et duree doivent etre strictement positifs.')
    source = base['sources']['acoustic_2d']
    if not isinstance(source['position'],list) or len(source['position'])!=2:
        raise ValueError('La configuration de reference doit avoir une source acoustique 2D unique.')
    cpp = repo/'src/IO_impl/json_parser_sources.cpp'
    multisource = repo/'src/callback_impl/Multisource.h'
    header = repo/'src/callback_impl/MovingDipole2D.hpp'
    if not cpp.is_file() or not multisource.is_file():
        raise ValueError('SPECFEM_DIR ne pointe pas vers le depot SPECFEM-Waves. '
                         'Adapter uniquement ce chemin en haut du script.')
    existing = cpp.read_text()
    target = existing
    if INCLUDE not in target:
        anchor = '#include "../callback_impl/Multisource.h"'
        if target.count(anchor)!=1:
            raise ValueError('Version du parseur non reconnue : aucun fichier modifie.')
        target = target.replace(anchor,anchor+'\n'+INCLUDE,1)
    if MARKER not in target:
        signature = 'std::enable_if_t<std::is_same_v<T<PHY>, SourceAcoustic2D<PHY>>, std::function<void(size_t, size_t)>>'
        if target.count(signature)!=1:
            raise ValueError('Fonction des sources 2D non reconnue : aucun fichier modifie.')
        start = target.index(signature)
        opening = target.index('\n{',start)+2
        target = target[:opening]+'\n'+BRANCH+target[opening:]
    elif BRANCH not in target:
        raise ValueError('Un autre correctif de source mobile est present : aucun fichier modifie.')
    if 'class MultiSourceBase2D' not in multisource.read_text():
        raise ValueError('Conteneur des sources 2D non reconnu : aucun fichier modifie.')
    if header.exists() and header.read_text()!=HEADER:
        raise ValueError('MovingDipole2D.hpp existe avec un autre contenu : aucun fichier modifie.')
    config = json.loads(json.dumps(base))
    config['nstep'] = int(math.ceil(duration/dt))
    config['sources'] = {'acoustic_2d':{
        'type':'moving_dipole',
        'position':[float(source['position'][0]),-PROFONDEUR_SOURCE],
        'amplitude':float(source['amplitude']),
        'frequency':float(source['frequency']),
        'speed_knots':VITESSE_NOEUDS,
        'spacing':1.0,
        'dipole_separation':1.0,
        'dipole_angle_deg':0.0,
        'ramp_seconds':0.1}}
    output = reference.with_name('config_mobile.json')
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get('sources',{}).get('acoustic_2d',{}).get('type')!='moving_dipole':
            raise ValueError('Un autre config_mobile.json existe : aucun fichier modifie.')
    # All compatibility checks are completed before any repository edit.
    if not header.exists():
        atomic_text(header,HEADER)
    if target!=existing:
        atomic_text(cpp,target)
    if not output.exists():
        atomic_text(output,json.dumps(config,indent=2)+'\n')
    else:
        config = json.loads(output.read_text())
        print('config_mobile.json deja present : tes reglages sont conserves.')
    head = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if head.returncode==0:
        print('Depot SPECFEM cible :',head.stdout.strip())
    t = config['dt']*config['nstep']
    s = config['sources']['acoustic_2d']
    v = s['speed_knots']*1852/3600
    print('Configuration generee :',output)
    print(f"Duree = {t:.9g} s ; dt = {config['dt']:g} ; nstep = {config['nstep']}")
    print(f"Source : x = {s['position'][0]:.6f} -> {s['position'][0]+v*t:.6f} m ; z = {s['position'][1]:g} m")
    print('Materiaux :',config['materials'])
    print('Recepteurs :',config['receivers'])
    print('Recompile le depot avec tes modules habituels, puis utilise config_mobile.json dans le job Slurm.')
    print('Aucun job lance. Les sorties conservent le chemin output de ta configuration validee.')
    return output


if __name__=='__main__':
    try:
        prepare(SPECFEM_DIR,CONFIG_REFERENCE)
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as error:
        raise SystemExit(str(error))
