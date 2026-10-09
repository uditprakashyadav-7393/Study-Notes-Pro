import os
import time
import re
import json
import base64
import binascii
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from supabase import create_client, Client


load_dotenv()


GEMINI_EMBEDDING_KEY = os.getenv("GEMINI_EMBEDDING_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


if not GEMINI_EMBEDDING_KEY:
    raise RuntimeError("GEMINI_EMBEDDING_KEY is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is missing")


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 1536

RAG_MATCH_THRESHOLD = 0.58
RAG_MATCH_COUNT = 3

MAX_HISTORY_MESSAGES = 6
MAX_NCERT_CONTEXT_CHARS = 5500
MAX_CHUNK_CHARS = 1100
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENT_TEXT_CHARS = 20000


gemini_embedding_client = genai.Client(
    api_key=GEMINI_EMBEDDING_KEY,
    http_options=genai.types.HttpOptions(timeout=90_000),
)

gemini_chat_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=genai.types.HttpOptions(timeout=90_000),
)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)


app = FastAPI(
    title="Study Notes Pro AI",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True
)


saved_chats: Dict[str, Dict[str, Any]] = {}


class Attachment(BaseModel):
    name: str
    type: Optional[str] = None
    size: Optional[int] = None
    content_base64: Optional[str] = None


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = ""
    class_level: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    chat_id: Optional[str] = None
    chat_title: Optional[str] = None
    history: List[HistoryMessage] = Field(
        default_factory=list
    )
    attachments: List[Attachment] = Field(
        default_factory=list
    )


CHAPTER_KEYWORDS = {

    "Physics" : {
    "Electric Charges and Fields": ["electric charges", "electric charge", "charge", "charges", "coulomb law", "coulombs law", "coulomb's law", "force between charges", "electric force", "electrostatic force", "superposition principle", "principle of superposition", "electric field", "electric field intensity", "field intensity", "electric field due to point charge", "field due to charge", "field due to charged sphere", "electric field due to dipole", "dipole field", "electric dipole", "electric dipole moment", "dipole moment", "axial position", "equatorial position", "electric field lines", "field lines", "properties of electric field lines", "electric flux", "flux", "electric flux density", "gauss law", "gauss's law", "gaussian surface", "gauss theorem", "applications of gauss law", "charged sphere", "charged shell", "charged conductor", "infinite wire", "infinite plane sheet", "uniformly charged sphere", "continuous charge distribution", "linear charge density", "surface charge density", "volume charge density", "lambda", "sigma", "rho", "electrostatics", "electrostatic field", "electric field numerical", "coulomb numerical", "gauss law numerical", "dipole numerical"],
    "Electrostatic Potential and Capacitance": ["electrostatic potential", "electric potential", "potential", "potential difference", "voltage", "potential due to point charge", "potential due to dipole", "potential energy", "electrostatic potential energy", "potential energy of system of charges", "equipotential surface", "equipotential", "properties of equipotential surface", "relation between electric field and potential", "electric field potential relation", "electrostatic equilibrium", "conductor in electrostatic equilibrium", "capacitance", "capacitor", "capacitors", "capacitance of capacitor", "parallel plate capacitor", "parallel plate condenser", "dielectric", "dielectric medium", "dielectric constant", "relative permittivity", "polarisation", "polarization of dielectric", "effect of dielectric", "combination of capacitors", "capacitors in series", "capacitors in parallel", "series combination", "parallel combination", "energy stored in capacitor", "energy density", "energy of capacitor", "charging capacitor", "discharging capacitor", "electric potential numerical", "capacitance numerical", "capacitor numerical"],
    "Current Electricity": ["current electricity", "electric current", "current", "flow of charge", "electric current density", "drift velocity", "drift speed", "mobility", "relaxation time", "electric current microscopic view", "ohm law", "ohm's law", "ohmic conductor", "non ohmic conductor", "v-i characteristics", "resistance", "electrical resistance", "resistivity", "specific resistance", "conductivity", "specific conductance", "temperature dependence of resistance", "temperature coefficient", "resistors", "combination of resistors", "resistors in series", "resistors in parallel", "emf", "emf of cell", "electromotive force", "internal resistance", "terminal voltage", "cells in series", "cells in parallel", "combination of cells", "kirchhoff laws", "kirchhoff's laws", "junction rule", "loop rule", "kirchhoff junction rule", "kirchhoff loop rule", "wheatstone bridge", "balanced wheatstone bridge", "meter bridge", "metre bridge", "potentiometer", "principle of potentiometer", "comparison of emf", "internal resistance using potentiometer", "electrical energy", "electrical power", "power dissipation", "current numerical", "resistance numerical", "kirchhoff numerical", "potentiometer numerical"],
    "Moving Charges and Magnetism": ["moving charges and magnetism", "moving charge", "magnetic field", "magnetic field intensity", "magnetic force", "force on moving charge", "lorentz force", "lorentz force equation", "charged particle in magnetic field", "motion of charged particle", "circular motion in magnetic field", "radius of circular path", "cyclotron frequency", "cyclotron", "biot savart law", "biot-savart law", "magnetic field due to current", "field due to straight wire", "long straight conductor", "field due to circular loop", "current carrying loop", "ampere law", "ampere's circuital law", "ampere circuital theorem", "solenoid", "magnetic field inside solenoid", "toroid", "magnetic field of toroid", "force on current carrying conductor", "force between parallel currents", "parallel current carrying wires", "definition of ampere", "current loop", "magnetic dipole", "magnetic dipole moment of current loop", "torque on current loop", "moving coil galvanometer", "galvanometer", "current sensitivity", "voltage sensitivity", "conversion of galvanometer", "galvanometer to ammeter", "galvanometer to voltmeter", "magnetic field numerical", "lorentz force numerical", "cyclotron numerical"],
    "Magnetism and Matter": ["magnetism and matter", "magnetism", "bar magnet", "bar magnet as magnetic dipole", "magnetic dipole", "magnetic dipole moment", "magnetic field of bar magnet", "magnetic field lines of magnet", "magnetic field axial point", "magnetic field equatorial point", "torque on magnetic dipole", "magnetic potential energy", "magnetic gauss law", "gauss law for magnetism", "earth magnetism", "earth's magnetic field", "magnetic elements of earth", "magnetic declination", "magnetic inclination", "horizontal component", "magnetisation", "magnetization", "magnetic intensity", "magnetic permeability", "relative permeability", "magnetic susceptibility", "diamagnetic", "diamagnetism", "paramagnetic", "paramagnetism", "ferromagnetic", "ferromagnetism", "magnetic materials", "magnetic properties of materials", "domain theory", "magnetic domains", "hysteresis", "hysteresis loop", "permanent magnets", "electromagnets", "soft iron", "hard magnetic material", "magnetic numerical"],
    "Electromagnetic Induction": ["electromagnetic induction", "emi", "electromagnetic induction chapter", "magnetic flux", "flux through coil", "change in magnetic flux", "faraday law", "faraday's law", "faraday laws of electromagnetic induction", "first law of faraday", "second law of faraday", "induced emf", "induced current", "induction", "electromagnetic induction numerical", "lenz law", "lenz's law", "direction of induced current", "fleming right hand rule", "motional emf", "emf due to motion", "moving rod in magnetic field", "magnetic flux change", "eddy currents", "eddy current applications", "self induction", "self-inductance", "coefficient of self induction", "mutual induction", "mutual inductance", "coefficient of mutual induction", "inductor", "inductance", "energy stored in inductor", "ac generator", "alternating current generator", "electric generator", "principle of generator", "induction coil", "transformer principle", "induced emf numerical", "faraday law numerical"],
    "Alternating Current": ["alternating current", "ac", "ac current", "alternating voltage", "ac voltage", "sinusoidal current", "sinusoidal voltage", "peak value", "maximum value", "instantaneous value", "average value", "rms value", "rms current", "rms voltage", "mean value of ac", "phasor", "phasor diagram", "ac through resistor", "ac through inductor", "ac through capacitor", "inductive reactance", "capacitive reactance", "reactance", "impedance", "resistance reactance impedance", "lcr circuit", "series lcr circuit", "rl circuit", "rc circuit", "lc circuit", "phase difference", "phase angle", "power in ac circuit", "ac power", "average power", "power factor", "wattless current", "resonance", "resonance in lcr", "resonant frequency", "quality factor", "transformer", "transformers", "step up transformer", "step down transformer", "transformer efficiency", "transformer losses", "eddy current loss", "ac numerical", "lcr numerical", "transformer numerical"],
    "Electromagnetic Waves": ["electromagnetic waves", "em waves", "em wave", "electromagnetic radiation", "displacement current", "maxwell displacement current", "electromagnetic spectrum", "spectrum of electromagnetic waves", "radio waves", "radio wave", "microwaves", "microwave", "infrared rays", "infrared radiation", "visible light", "visible spectrum", "ultraviolet rays", "ultraviolet radiation", "x rays", "x-rays", "x ray", "gamma rays", "gamma radiation", "frequency of electromagnetic waves", "wavelength of electromagnetic waves", "speed of electromagnetic waves", "properties of electromagnetic waves", "uses of radio waves", "uses of microwaves", "uses of infrared", "uses of ultraviolet", "uses of x rays", "uses of gamma rays", "electromagnetic wave numerical"],
    "Ray Optics and Optical Instruments": ["ray optics", "ray optics and optical instruments", "geometrical optics", "light", "reflection of light", "laws of reflection", "refraction of light", "laws of refraction", "snell law", "snell's law", "refractive index", "relative refractive index", "apparent depth", "real depth", "spherical mirror", "spherical mirrors", "concave mirror", "convex mirror", "pole", "centre of curvature", "radius of curvature", "principal axis", "principal focus", "focal length", "mirror formula", "magnification of mirror", "mirror numerical", "lens", "lenses", "convex lens", "concave lens", "thin lens", "lens formula", "lens maker formula", "lens maker's formula", "magnification of lens", "power of lens", "combination of lenses", "lenses in contact", "prism", "refraction through prism", "angle of deviation", "minimum deviation", "total internal reflection", "tir", "critical angle", "optical fibre", "optical fiber", "human eye", "eye defects", "myopia", "hypermetropia", "presbyopia", "correction of eye defects", "microscope", "compound microscope", "telescope", "astronomical telescope", "magnifying power", "optical instruments numerical", "lens numerical", "mirror numerical"],
    "Wave Optics": ["wave optics", "wave nature of light", "wavefront", "wave front", "spherical wavefront", "plane wavefront", "cylindrical wavefront", "huygens principle", "huygens' principle", "huygens wave theory", "reflection using huygens principle", "refraction using huygens principle", "interference", "interference of light", "constructive interference", "destructive interference", "coherent sources", "coherence", "young double slit experiment", "young's double slit experiment", "ydse", "double slit experiment", "fringe", "fringe width", "bright fringe", "dark fringe", "path difference", "phase difference", "intensity distribution", "diffraction", "diffraction of light", "single slit diffraction", "width of central maximum", "polarisation", "polarization", "plane polarised light", "polaroids", "malus law", "wave optics numerical", "ydse numerical", "fringe width numerical"],
    "Dual Nature of Radiation and Matter": ["dual nature of radiation and matter", "dual nature", "dual nature of light", "particle nature of light", "wave particle duality", "photoelectric effect", "photoelectric emission", "photoelectric phenomenon", "photoelectric current", "photoelectric cell", "photoelectric equation", "einstein photoelectric equation", "einstein equation", "work function", "threshold frequency", "threshold wavelength", "stopping potential", "maximum kinetic energy", "photoelectric graph", "frequency vs kinetic energy", "intensity vs photoelectric current", "laws of photoelectric emission", "photocurrent", "photon", "photons", "energy of photon", "momentum of photon", "matter waves", "de broglie hypothesis", "de broglie wavelength", "de broglie equation", "wave nature of matter", "electron wavelength", "dual nature numerical", "photoelectric numerical", "de broglie numerical"],
    "Atoms": ["atoms", "atom", "atomic physics", "rutherford model", "rutherford atomic model", "rutherford scattering experiment", "alpha particle scattering", "nuclear model of atom", "limitations of rutherford model", "bohr model", "bohr atomic model", "bohr postulates", "bohr's postulates", "bohr radius", "radius of orbit", "energy of electron", "energy levels", "energy level diagram", "quantised energy levels", "hydrogen atom", "hydrogen spectrum", "atomic spectrum", "line spectrum", "emission spectrum", "absorption spectrum", "spectral series", "balmer series", "lyman series", "paschen series", "brackett series", "pfund series", "rydberg formula", "rydberg equation", "excitation energy", "ionisation energy", "ionization energy", "ground state", "excited state", "transition of electron", "photon emission", "photon absorption", "atomic model numerical", "bohr model numerical"],
    "Nuclei": ["nuclei", "nucleus", "nuclear physics", "atomic nucleus", "nuclear structure", "nuclear size", "nuclear radius", "nuclear mass", "nuclear density", "atomic mass unit", "amu", "isotopes", "isobars", "isotones", "mass defect", "mass defect formula", "binding energy", "binding energy per nucleon", "binding energy curve", "nuclear force", "properties of nuclear force", "strong nuclear force", "radioactivity", "radioactive decay", "radioactive substance", "decay constant", "decay law", "activity", "radioactive activity", "half life", "half-life", "mean life", "average life", "alpha decay", "alpha emission", "beta decay", "beta emission", "gamma decay", "gamma emission", "radioactive series", "nuclear reaction", "nuclear equation", "mass energy relation", "einstein mass energy relation", "nuclear fission", "fission", "nuclear fusion", "fusion", "chain reaction", "critical mass", "atomic bomb", "hydrogen bomb", "nuclear reactor", "nuclear energy", "nuclear numerical", "radioactivity numerical", "binding energy numerical"],
    "Semiconductor Electronics: Materials, Devices and Simple Circuits": ["semiconductor electronics", "semiconductor", "semiconductor materials", "electronic devices", "conductor semiconductor insulator", "energy bands", "valence band", "conduction band", "forbidden energy gap", "energy gap", "intrinsic semiconductor", "extrinsic semiconductor", "doping", "doped semiconductor", "p type semiconductor", "p-type semiconductor", "n type semiconductor", "n-type semiconductor", "majority carriers", "minority carriers", "electron hole pair", "holes", "p-n junction", "pn junction", "junction diode", "semiconductor diode", "depletion region", "depletion layer", "potential barrier", "barrier potential", "forward bias", "reverse bias", "diode characteristics", "v-i characteristics", "diode equation", "rectifier", "rectification", "half wave rectifier", "full wave rectifier", "bridge rectifier", "filter circuit", "zener diode", "zener breakdown", "zener voltage", "voltage regulator", "led", "light emitting diode", "photodiode", "solar cell", "photovoltaic cell", "logic gates", "digital electronics", "and gate", "or gate", "not gate", "nand gate", "nor gate", "truth table", "boolean logic", "logic gate symbols", "semiconductor numerical", "diode numerical"],
    },
    "Mathematics" : {
    "Relations and Functions": ['maths ch1 ', "Maths chapter 1", "Mathematics chapter 1 ", "Mathematics ch 1", "relations and functions", "relation", "relations", "functions", "types of relations", "reflexive relation", "symmetric relation", "transitive relation", "equivalence relation", "equivalence relations", "empty relation", "universal relation", "identity relation", "inverse relation", "domain", "range", "codomain", "types of functions", "one one function", "one-one function", "many one function", "many-one function", "into function", "onto function", "one-one onto function", "bijective function", "composite function", "composition of functions", "invertible function", "inverse function", "inverse of function", "binary operation", "binary operations", "properties of binary operation", "closure", "commutative", "associative", "identity element", "inverse element", "function numerical", "relation numerical"],
    "Inverse Trigonometric Functions": ['maths ch 2 ', "Maths chapter 2", "Mathematics chapter 2 ", "Mathematics ch 2", "inverse trigonometric functions", "inverse trigonometry", "inverse trigonometric function", "inverse sine", "sin inverse", "sin^-1", "arcsin", "inverse cosine", "cos inverse", "cos^-1", "arccos", "inverse tangent", "tan inverse", "tan^-1", "arctan", "inverse cot", "cot inverse", "inverse sec", "sec inverse", "inverse cosec", "cosec inverse", "principal value", "principal values", "principal branch", "domain of inverse trigonometric functions", "range of inverse trigonometric functions", "properties of inverse trigonometric functions", "identities of inverse trigonometric functions", "inverse trigonometric identities", "trigonometric inverse identities", "inverse trigonometry numerical"],
    "Matrices": ['maths ch 3 ', "Maths chapter 3", "Mathematics chapter 3 ", "Mathematics ch 3", "matrices", "matrix", "types of matrices", "row matrix", "column matrix", "rectangular matrix", "square matrix", "zero matrix", "null matrix", "diagonal matrix", "scalar matrix", "identity matrix", "unit matrix", "triangular matrix", "upper triangular matrix", "lower triangular matrix", "symmetric matrix", "skew symmetric matrix", "matrix order", "order of matrix", "elements of matrix", "matrix notation", "equality of matrices", "addition of matrices", "subtraction of matrices", "multiplication of matrices", "scalar multiplication", "matrix multiplication", "properties of matrix multiplication", "transpose", "transpose of matrix", "properties of transpose", "symmetric matrix", "skew symmetric matrix", "inverse of matrix", "matrix inverse", "elementary operations", "elementary row operations", "elementary column operations", "matrix equation", "matrix numerical"],
    "Determinants": ["determinants", "determinant", "determinant of matrix", "determinant of order 2", "determinant of order 3", "properties of determinants", "value of determinant", "minor", "minors", "cofactor", "cofactors", "cofactor expansion", "expansion of determinant", "area of triangle", "area using determinant", "adjoint", "adjoint of matrix", "inverse using adjoint", "inverse using determinant", "singular matrix", "non singular matrix", "system of linear equations", "linear equations", "solution of linear equations", "consistent system", "inconsistent system", "unique solution", "no solution", "infinitely many solutions", "cramer rule", "cramer's rule", "determinant numerical"],
    "Continuity and Differentiability": ["continuity and differentiability", "continuity", "continuous function", "discontinuous function", "continuity at a point", "continuity on interval", "left hand limit", "right hand limit", "left hand derivative", "right hand derivative", "differentiability", "differentiable function", "derivative", "derivatives", "differentiation", "derivative from first principle", "first principle", "standard derivatives", "derivative of polynomial", "derivative of trigonometric function", "derivative of exponential function", "derivative of logarithmic function", "derivative of inverse trigonometric function", "chain rule", "product rule", "quotient rule", "implicit differentiation", "logarithmic differentiation", "parametric differentiation", "second order derivative", "higher order derivative", "mean value theorem", "mean value theorems", "rolle theorem", "rolle's theorem", "lagrange mean value theorem", "lagrange's mean value theorem", "lmvt", "continuity numerical", "differentiability numerical"],
    "Applications of Derivatives": ["applications of derivatives", "application of derivatives", "rate of change", "rate of change of quantities", "related rates", "increasing function", "decreasing function", "increasing and decreasing functions", "monotonic function", "monotonicity", "strictly increasing", "strictly decreasing", "first derivative test", "critical point", "critical points", "critical number", "local maximum", "local minimum", "maxima", "minima", "maximum value", "minimum value", "absolute maximum", "absolute minimum", "optimization", "optimization problems", "maximum and minimum", "tangent", "normal", "slope", "derivative application", "increasing decreasing numerical", "maxima minima numerical"],
    "Integrals": ["integrals", "integral", "integration", "indefinite integral", "definite integral", "integration as inverse process", "anti derivative", "antiderivative", "standard integrals", "basic integration formulas", "integration by substitution", "substitution method", "integration by parts", "by parts", "partial fractions", "integration using partial fractions", "integration of trigonometric functions", "trigonometric integrals", "integrals involving logarithm", "integrals involving exponential", "integrals involving inverse trigonometric functions", "definite integral properties", "properties of definite integrals", "fundamental theorem of calculus", "first fundamental theorem", "second fundamental theorem", "upper limit", "lower limit", "variable limit", "integration numerical", "definite integral numerical", "indefinite integral numerical", "evaluate integral", "find integral"],
    "Applications of Integrals": ["applications of integrals", "application of integrals", "area under curve", "area under the curve", "area bounded by curve", "area bounded by curves", "area between curves", "area enclosed by curves", "area between two curves", "area of region", "area using integration", "geometrical interpretation of integral", "area under x axis", "area above x axis", "area with respect to x", "area with respect to y", "area of circle", "area of parabola", "area of ellipse", "area between line and curve", "area between two lines", "area between parabola and line", "definite integration area", "area numerical"],
    "Differential Equations": ["differential equations", "differential equation", "equation involving derivatives", "order of differential equation", "degree of differential equation", "order and degree", "general solution", "particular solution", "solution of differential equation", "formation of differential equation", "formation of differential equations", "variable separable method", "separation of variables", "separable differential equation", "homogeneous differential equation", "homogeneous equation", "linear differential equation", "first order differential equation", "first degree differential equation", "dy dx", "dy/dx", "dx dy", "differential equation numerical", "solve differential equation", "formation of differential equation numerical"],
    "Vector Algebra": ["vector algebra", "vectors", "vector", "scalar", "vector quantities", "magnitude of vector", "modulus of vector", "unit vector", "zero vector", "null vector", "equal vectors", "negative vector", "position vector", "co initial vectors", "parallel vectors", "collinear vectors", "coplanar vectors", "direction cosines", "direction ratios", "section formula", "section formula vector", "addition of vectors", "subtraction of vectors", "scalar multiplication", "dot product", "scalar product", "vector product", "cross product", "properties of dot product", "properties of cross product", "projection of vector", "scalar projection", "vector projection", "angle between vectors", "perpendicular vectors", "parallel vectors", "area of parallelogram", "area of triangle using vectors", "scalar triple product", "vector triple product", "vector numerical"],
    "Three Dimensional Geometry": ["three dimensional geometry", "3d geometry", "three dimensional", "3 dimensional geometry", "coordinate geometry in space", "coordinates in space", "direction cosines", "direction ratios", "line in three dimensions", "straight line in 3d", "equation of line", "line equation", "vector equation of line", "cartesian equation of line", "symmetric equation of line", "parametric equation of line", "direction ratios of line", "angle between two lines", "parallel lines", "perpendicular lines", "skew lines", "coplanar lines", "shortest distance", "shortest distance between lines", "distance between skew lines", "distance between two lines", "plane", "equation of plane", "vector equation of plane", "cartesian equation of plane", "normal vector", "angle between planes", "angle between line and plane", "distance of point from plane", "distance between parallel planes", "3d numerical", "line numerical", "plane numerical"],
    "Linear Programming": ["linear programming", "linear programming problem", "lpp", "linear programming problems", "objective function", "constraints", "constraints in lpp", "linear constraints", "non negative constraints", "feasible region", "feasible solution", "feasible solutions", "optimal solution", "optimal value", "optimal point", "corner point", "corner point method", "graphical method", "graphical solution", "maximisation", "maximization", "minimisation", "minimization", "maximize", "minimize", "bounded region", "unbounded region", "bounded feasible region", "unbounded feasible region", "infeasible problem", "linear inequalities", "graph of inequalities", "lpp numerical", "linear programming numerical"],
    "Probability": ["probability", "probabilities", "conditional probability", "conditional probabilities", "conditional probability formula", "independent events", "independent event", "dependent events", "multiplication theorem", "multiplication rule", "addition theorem", "bayes theorem", "bayes theorem of probability", "bayes rule", "random variable", "random variables", "discrete random variable", "continuous random variable", "probability distribution", "probability distribution table", "probability mass function", "mean of random variable", "expectation", "mathematical expectation", "variance", "standard deviation", "binomial distribution", "binomial probability distribution", "binomial random variable", "bernoulli trials", "bernoulli experiment", "success and failure", "binomial theorem probability", "probability numerical", "conditional probability numerical", "bayes theorem numerical", "binomial distribution numerical"],
    },
    "Chemistry" : {
    "Solutions": ["solutions", "solution", "types of solutions", "solute", "solvent", "concentration", "molarity", "molality", "mole fraction", "mass percentage", "volume percentage", "ppm", "parts per million", "solubility", "solubility of gases", "Henry's law", "Raoult's law", "vapour pressure", "relative lowering of vapour pressure", "elevation in boiling point", "boiling point elevation", "depression in freezing point", "freezing point depression", "osmotic pressure", "osmosis", "reverse osmosis", "semipermeable membrane", "colligative properties", "abnormal molar mass", "van't Hoff factor", "association", "dissociation", "numericals on solutions", "solution numericals", "molarity questions", "molality questions", "colligative property numericals"],
    "Electrochemistry": ["electrochemistry", "electrochemical cell", "electrolytic cell", "galvanic cell", "voltaic cell", "electrode", "anode", "cathode", "electrolyte", "salt bridge", "cell potential", "electrode potential", "standard electrode potential", "standard reduction potential", "EMF", "cell emf", "Nernst equation", "Nernst", "Gibbs energy", "free energy", "delta G", "equilibrium constant", "conductance", "resistance", "conductivity", "molar conductivity", "specific conductivity", "Kohlrausch law", "Kohlrausch's law", "variation of conductivity", "electrolysis", "Faraday laws", "Faraday's first law", "Faraday's second law", "electrochemical series", "batteries", "dry cell", "lead storage battery", "fuel cell", "corrosion", "rusting", "electrochemical corrosion", "electrochemistry numericals", "Nernst equation numericals", "conductivity numericals"],
    "Chemical Kinetics": ["chemical kinetics", "rate of reaction", "reaction rate", "average rate", "instantaneous rate", "rate law", "rate constant", "order of reaction", "molecularity", "zero order reaction", "first order reaction", "second order reaction", "integrated rate equation", "zero order equation", "first order equation", "half life", "half-life", "pseudo first order reaction", "Arrhenius equation", "activation energy", "threshold energy", "collision theory", "temperature dependence", "frequency factor", "rate constant units", "reaction mechanism", "rate determining step", "kinetics numericals", "order numericals", "half life numericals", "Arrhenius numericals"],
    "d and f Block Elements": ["d and f block elements", "d block", "f block", "transition elements", "inner transition elements", "transition metals", "lanthanides", "actinides", "electronic configuration", "variable oxidation states", "oxidation states", "atomic radii", "ionic radii", "ionization enthalpy", "magnetic properties", "paramagnetic", "diamagnetic", "colour of compounds", "catalytic properties", "complex formation", "alloy formation", "interstitial compounds", "lanthanide contraction", "consequences of lanthanide contraction", "actinide contraction", "KMnO4", "potassium permanganate", "K2Cr2O7", "potassium dichromate", "oxidising properties", "redox reactions", "transition metal compounds", "d block questions", "f block questions", "lanthanides questions", "actinides questions"],
    "Coordination Compounds": ["coordination compounds", "coordination chemistry", "coordination entity", "complex compound", "complex ion", "coordination number", "coordination sphere", "central metal atom", "central metal ion", "ligand", "monodentate ligand", "bidentate ligand", "polydentate ligand", "ambidentate ligand", "chelating ligand", "coordination polyhedron", "oxidation number", "coordination nomenclature", "IUPAC nomenclature", "naming coordination compounds", "Werner theory", "Werner's coordination theory", "VBT", "valence bond theory", "CFT", "crystal field theory", "crystal field splitting", "d orbitals splitting", "octahedral complex", "tetrahedral complex", "square planar complex", "magnetic properties", "paramagnetic complex", "diamagnetic complex", "colour of coordination compounds", "isomerism", "structural isomerism", "stereoisomerism", "geometrical isomerism", "optical isomerism", "coordination isomerism", "ionisation isomerism", "linkage isomerism", "applications of coordination compounds"],
    "Haloalkanes and Haloarenes": ["haloalkanes", "haloarenes", "alkyl halides", "aryl halides", "halogen derivatives", "C-X bond", "carbon halogen bond", "preparation of haloalkanes", "preparation of haloarenes", "nucleophilic substitution", "SN1", "SN2", "SN1 reaction", "SN2 reaction", "elimination reaction", "dehydrohalogenation", "Wurtz reaction", "Finkelstein reaction", "Swarts reaction", "Sandmeyer reaction", "Gattermann reaction", "Balz Schiemann reaction", "Grignard reagent", "Grignard reaction", "organomagnesium compound", "physical properties", "chemical properties", "optical activity", "stereochemistry", "reaction mechanism", "haloalkane questions", "haloarene questions", "SN1 vs SN2"],
    "Alcohols Phenols and Ethers": ["alcohols", "phenols", "ethers", "alcohol", "phenol", "ether", "classification of alcohols", "primary alcohol", "secondary alcohol", "tertiary alcohol", "monohydric alcohol", "dihydric alcohol", "trihydric alcohol", "preparation of alcohols", "preparation of phenols", "preparation of ethers", "properties of alcohols", "properties of phenols", "properties of ethers", "dehydration of alcohol", "oxidation of alcohol", "Lucas test", "phenol acidity", "Kolbe reaction", "Kolbe's reaction", "Reimer Tiemann reaction", "Reimer-Tiemann", "Williamson ether synthesis", "ether cleavage", "HI reaction", "alcohol reactions", "phenol reactions", "ether reactions"],
    "Aldehydes Ketones and Carboxylic Acids": ["aldehydes", "ketones", "carboxylic acids", "carbonyl compounds", "carbonyl group", "aldehyde group", "ketone group", "carboxyl group", "preparation of aldehydes", "preparation of ketones", "preparation of carboxylic acids", "nucleophilic addition", "addition reaction", "oxidation", "reduction", "Tollens test", "Tollen's test", "Fehling test", "Fehling's solution", "iodoform test", "2,4-DNP test", "Aldol condensation", "aldol reaction", "Cannizzaro reaction", "Clemmensen reduction", "Wolff Kishner reduction", "Rosenmund reduction", "Stephen reaction", "Hell Volhard Zelinsky", "HVZ reaction", "acidity of carboxylic acids", "aldehyde ketone reactions", "carbonyl reactions", "name reactions", "organic conversion", "distinguish aldehyde ketone"],
    "Amines": ["amines", "amine", "primary amine", "secondary amine", "tertiary amine", "aliphatic amine", "aromatic amine", "aniline", "classification of amines", "preparation of amines", "properties of amines", "basicity of amines", "basic strength", "Hofmann bromamide reaction", "Hoffmann bromamide", "Gabriel phthalimide synthesis", "Gabriel synthesis", "reduction of nitro compounds", "diazonium salts", "diazotisation", "diazonium salt preparation", "Sandmeyer reaction", "Gattermann reaction", "azo coupling", "coupling reaction", "carbylamine reaction", "Hinsberg test", "Hinsberg reagent", "aniline reactions", "amine reactions", "amines nomenclature", "amines questions"],
    "Biomolecules": ["biomolecules", "carbohydrates", "monosaccharides", "disaccharides", "polysaccharides", "glucose", "fructose", "sucrose", "maltose", "lactose", "starch", "cellulose", "glycogen", "reducing sugar", "non reducing sugar", "proteins", "amino acids", "peptide bond", "polypeptide", "protein structure", "primary structure", "secondary structure", "tertiary structure", "quaternary structure", "denaturation", "enzymes", "biological catalysts", "vitamins", "fat soluble vitamins", "water soluble vitamins", "vitamin A", "vitamin B", "vitamin C", "vitamin D", "vitamin E", "vitamin K", "nucleic acids", "DNA", "RNA", "nucleotides", "nucleosides", "biomolecules questions"],
},
    "Computer-Science" :{
    "Exception Handling in Python": ["exception handling", "exception", "error handling", "errors", "runtime error", "syntax error", "logical error", "try", "except", "else", "finally", "raise", "try block", "except block", "finally block", "multiple exceptions", "nested try", "built in exceptions", "user defined exception", "ZeroDivisionError", "ValueError", "TypeError", "IndexError", "KeyError", "NameError", "FileNotFoundError", "ImportError", "exception object", "handling errors", "exception program", "exception output", "exception questions", "exception MCQ", "python exception"],
    "File Handling in Python": ["file handling", "file handling in python", "files in python", "file", "text file", "binary file", "CSV file", "file object", "file operations", "open", "close", "read", "write", "append", "readline", "readlines", "writelines", "seek", "tell", "file pointer", "file modes", "r mode", "w mode", "a mode", "r+ mode", "w+ mode", "a+ mode", "rb mode", "wb mode", "ab mode", "with statement", "text file handling", "binary file handling", "CSV handling", "csv module", "csv reader", "csv writer", "reader", "writer", "writerow", "writerows", "pickle", "pickle module", "pickle dump", "pickle load", "dump", "load", "file search", "file update", "file append", "file program", "file output", "file questions", "file handling MCQ"],
    "Stack": ["stack","stack data structure","stack in python","LIFO","last in first out","stack operations","push","pop","peek","top","top element","push operation","pop operation","stack insertion","stack deletion","stack traversal","stack implementation","stack using list","stack overflow","stack underflow","empty stack","stack algorithm","stack program","stack questions","stack output","stack MCQ"],
    "Queue": ["queue","queue data structure","queue in python","FIFO","first in first out","queue operations","enqueue","dequeue","front","rear","front element","rear element","queue insertion","queue deletion","queue traversal","queue implementation","queue using list","linear queue","circular queue","queue overflow","queue underflow","empty queue","queue algorithm","queue program","queue questions","queue output","queue MCQ"],
    "Sorting": ["sorting","sorting algorithms","sorting technique","sort","ascending order","descending order","sorting list","sorting array","bubble sort","bubble sort algorithm","selection sort","selection sort algorithm","insertion sort","insertion sort algorithm","merge sort","quick sort","comparison sorting","stable sorting","in-place sorting","swapping","sorting steps","sorting algorithm complexity","sorting program","sorting output","sort a list","sort elements","sorting questions","sorting MCQ"],
    "Searching": ["searching","searching algorithms","search technique","search","linear search","sequential search","binary search","linear search algorithm","binary search algorithm","search element","search key","key element","sorted list","unsorted list","searching in list","searching in array","low","high","middle","mid","search position","successful search","unsuccessful search","search algorithm","search program","search output","search questions","searching MCQ","linear vs binary search"],
    "Understanding Data": ["understanding data","data","data and information","information","dataset","data set","data types","structured data","unstructured data","semi structured data","data collection","data representation","data analysis","data processing","data visualization","data interpretation","data patterns","data trends","data cleaning","data quality","data source","primary data","secondary data","categorical data","numerical data","qualitative data","quantitative data","mean","median","mode","average","range","frequency","frequency distribution","charts","graphs","bar graph","line graph","pie chart","histogram","data questions","data analysis questions"],
    "Database Concept": ["database concept","database concepts","database","DBMS","database management system","RDBMS","relational database","database system","data","table","record","row","column","field","attribute","tuple","relation","domain","database schema","database instance","primary key","candidate key","alternate key","foreign key","composite key","key","relationship","database model","relational model","database design","data redundancy","data inconsistency","data integrity","database security","database advantages","database applications","DBMS advantages","DBMS questions","database MCQ"],
    "SQL": ["SQL","structured query language","SQL query","SQL commands","SQL statements","database query","DDL","DML","CREATE","CREATE DATABASE","CREATE TABLE","ALTER","ALTER TABLE","DROP","DROP TABLE","INSERT","INSERT INTO","UPDATE","DELETE","SELECT","WHERE","DISTINCT","ORDER BY","GROUP BY","BETWEEN","NULL","IS NULL","IS NOT NULL","SQL data types","INT","INTEGER","FLOAT","DECIMAL","CHAR","VARCHAR","DATE","primary key","foreign key","NOT NULL","UNIQUE","DEFAULT","COUNT","SUM","AVG","MIN","MAX","aggregate functions","SQL functions","join","SQL joins","cartesian product","equi join","natural join","SQL output","SQL program","SQL practice","SQL questions","SQL MCQ"],
    "Computer Network": ["computer network","computer networks","network","networking","network communication","network architecture","network devices","hub","switch","router","repeater","bridge","gateway","modem","NIC","network interface card","network topology","bus topology","star topology","ring topology","tree topology","mesh topology","LAN","MAN","WAN","PAN","local area network","metropolitan area network","wide area network","personal area network","internet","intranet","network protocol","TCP","TCP/IP","HTTP","HTTPS","FTP","SMTP","POP3","IP address","IPv4","IPv6","MAC address","URL","domain name","WWW","World Wide Web","web server","web browser","network security","firewall","network questions","network MCQ"],
    "Data Communication": ["data communication","communication","data transmission","sender","receiver","message","communication channel","transmission medium","protocol","data signal","analog signal","digital signal","bandwidth","data transfer rate","bit rate","baud rate","transmission mode","simplex","half duplex","full duplex","serial transmission","parallel transmission","synchronous transmission","asynchronous transmission","wired communication","wireless communication","twisted pair cable","coaxial cable","optical fiber","optical fibre","radio waves","microwave","infrared","satellite communication","communication devices","modem","multiplexing","data communication model","communication protocol","data communication questions","data communication MCQ"],
    "Security Aspects": ["security aspects","computer security","cyber security","cyber safety","information security","data security","network security","privacy","data privacy","cyber crime","cybercrime","hacking","phishing","spoofing","identity theft","online fraud","malware","virus""worm","trojan horse","ransomware","spyware","adware","antivirus","firewall","password security","strong password","authentication","authorization","encryption","decryption","digital signature","OTP","two factor authentication","2FA","social engineering","spam","cyber ethics","digital footprint","cyberbullying","safe browsing","privacy protection","IT Act","cyber law","intellectual property","copyright","plagiarism","software piracy","e waste","electronic waste","green computing","security questions","cyber security MCQ"],
    "Project Based Learning": ["project based learning","project","project work","computer science project","CS project","project development","project planning","problem identification","problem definition","requirements","data collection","data analysis","algorithm","flowchart","program development","Python project","database project","SQL project","testing","debugging","implementation","documentation","project report","project presentation","project viva","project synopsis","project methodology","project objectives","project outcome","project evaluation","case study","real world project","application development"]
}
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\+\./']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def call_gemini(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
):
    """Send the prompt and conversation turns to the Gemini chat model."""
    system_instruction = "\n\n".join(
        message["content"].strip()
        for message in messages
        if message.get("role") == "system"
        and message.get("content", "").strip()
    )
    if not system_instruction:
        raise ValueError("Gemini requests require a system instruction.")

    contents = [
        {
            "role": "model" if message.get("role") == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and message.get("content", "").strip()
    ]

    config_options = {
        "system_instruction": system_instruction,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if response_mime_type:
        config_options["response_mime_type"] = response_mime_type

    return gemini_chat_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai.types.GenerateContentConfig(**config_options),
    )


def extract_attachment_text(attachment: Attachment) -> str:
    """Extract text from uploaded PDFs and common UTF-8 text files."""
    if not attachment.content_base64:
        return ""

    try:
        content = base64.b64decode(
            attachment.content_base64,
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Attachment '{attachment.name}' is not valid base64.",
        ) from error

    if len(content) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Attachment '{attachment.name}' exceeds the 5 MB limit.",
        )

    content_type = (attachment.type or "").split(";", 1)[0].strip().lower()
    file_name = attachment.name.lower()

    if content_type == "application/pdf" or file_name.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{attachment.name}' is not a readable PDF.",
            ) from error
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Attachment '{attachment.name}' is not a supported text file. "
                    "Upload a PDF or UTF-8 text file."
                ),
            ) from error

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"No readable text was found in '{attachment.name}'.",
        )

    return text[:MAX_ATTACHMENT_TEXT_CHARS]


def extract_requested_question_count(
    user_message: str,
    default: int = 10,
) -> int:
    """Extract the requested quiz size from chat text, with a safe cap."""
    normalized = normalize_text(user_message or "")
    if not normalized:
        return max(1, min(default, 30))

    match = re.search(
        r"\b(\d+)\s*(?:mcq|mcqs|multiple choice questions?|questions?|quiz)\b",
        normalized,
    )

    if not match:
        return max(1, min(default, 30))

    requested_count = int(match.group(1))
    return max(1, min(requested_count, 30))


def get_quiz_max_tokens(requested_count: int) -> int:
    """Give larger quizzes enough output room without blowing past reasonable limits."""
    requested_count = max(1, min(requested_count, 30))
    return min(12000, max(6000, 2500 + requested_count * 350))


def generate_fallback_mcq_quiz(
    class_level: str,
    subject: Optional[str],
    chapter: Optional[str],
    user_message: str,
) -> Dict[str, Any]:
    """Deterministic MCQ fallback used when the model fails or returns malformed JSON."""
    requested_count = extract_requested_question_count(user_message)
    subject_text = (subject or "General").strip() or "General"
    physics_questions = [
        {
            "question": "The SI unit of electric charge is:",
            "options": ["Coulomb", "Volt", "Ampere", "Ohm"],
            "answer": "A",
            "hint": "Charge is measured in coulombs.",
            "explanation": "Electric charge is measured in coulombs, which is the SI unit of charge."
        },
        {
            "question": "Ohm's law relates:",
            "options": ["Voltage, current and resistance", "Mass, force and acceleration", "Energy, power and time", "Charge, field and flux"],
            "answer": "A",
            "hint": "It is the formula V = IR.",
            "explanation": "Ohm's law states that V = IR, so voltage, current and resistance are related."
        },
        {
            "question": "Which device is used to measure electric current?",
            "options": ["Ammeter", "Voltmeter", "Galvanometer", "Thermometer"],
            "answer": "A",
            "hint": "It is connected in series in a circuit.",
            "explanation": "An ammeter is used to measure current and is connected in series."
        },
        {
            "question": "The force between two like charges is:",
            "options": ["Repulsive", "Attractive", "Zero", "Variable"],
            "answer": "A",
            "hint": "Like charges repel each other.",
            "explanation": "Like charges repel; unlike charges attract."
        },
        {
            "question": "Which quantity is measured in joules?",
            "options": ["Energy", "Resistance", "Current", "Charge"],
            "answer": "A",
            "hint": "Energy is the capacity to do work.",
            "explanation": "The SI unit of energy is the joule."
        },
        {
            "question": "The magnetic field inside a long solenoid is:",
            "options": ["Uniform", "Zero", "Radial", "Non-uniform"],
            "answer": "A",
            "hint": "The field is nearly same at every point inside the solenoid.",
            "explanation": "Inside a long solenoid, the magnetic field is nearly uniform."
        },
        {
            "question": "The SI unit of resistance is:",
            "options": ["Ohm", "Coulomb", "Tesla", "Watt"],
            "answer": "A",
            "hint": "It is symbolized by the Greek letter omega.",
            "explanation": "Resistance is measured in ohms (Ω)."
        },
        {
            "question": "A conductor allows current to pass because it has:",
            "options": ["Free electrons", "Fixed protons", "High resistance", "No charge"],
            "answer": "A",
            "hint": "Electrons move freely in metals.",
            "explanation": "Metals have free electrons that move and carry current."
        },
        {
            "question": "The work done in moving a unit positive charge from one point to another is called:",
            "options": ["Potential difference", "Current", "Power", "Resistance"],
            "answer": "A",
            "hint": "It is the difference in electric potential between two points.",
            "explanation": "Potential difference is the work done per unit charge between two points."
        },
        {
            "question": "Faraday's law is related to:",
            "options": ["Electromagnetic induction", "Ohm's law", "Conservation of mass", "Photoelectric effect"],
            "answer": "A",
            "hint": "It explains induced emf due to changing magnetic flux.",
            "explanation": "Faraday's law explains how a changing magnetic flux induces emf."
        },
        {
            "question": "The speed of light in vacuum is approximately:",
            "options": ["3 × 10^8 m/s", "3 × 10^5 m/s", "3 × 10^2 m/s", "3 × 10^10 m/s"],
            "answer": "A",
            "hint": "It is a very large value in SI units.",
            "explanation": "Light travels at approximately 3 × 10^8 m/s in vacuum."
        },
        {
            "question": "Which of the following is a scalar quantity?",
            "options": ["Electric charge", "Velocity", "Force", "Acceleration"],
            "answer": "A",
            "hint": "It has magnitude only.",
            "explanation": "Electric charge is a scalar quantity; velocity, force, and acceleration are vectors."
        },
        {
            "question": "In a series circuit, current is:",
            "options": ["Same through every component", "Different in each component", "Always zero", "Always increasing"],
            "answer": "A",
            "hint": "There is only one path for current.",
            "explanation": "In a series circuit, the same current flows through all components."
        },
        {
            "question": "The SI unit of magnetic field is:",
            "options": ["Tesla", "Weber", "Coulomb", "Newton"],
            "answer": "A",
            "hint": "It is named after Nikola Tesla.",
            "explanation": "Magnetic field strength is measured in tesla (T)."
        },
        {
            "question": "Potential difference is measured by a:",
            "options": ["Voltmeter", "Ammeter", "Barometer", "Hydrometer"],
            "answer": "A",
            "hint": "It is connected in parallel across the component.",
            "explanation": "A voltmeter measures potential difference and is connected in parallel."
        },
        {
            "question": "The energy stored in a capacitor is proportional to:",
            "options": ["The square of voltage", "The square of current", "The resistance", "The time"],
            "answer": "A",
            "hint": "Use the formula U = 1/2 CV^2.",
            "explanation": "The energy stored is U = 1/2 CV^2, so it depends on V^2."
        },
        {
            "question": "A current-carrying conductor experiences force in a magnetic field according to:",
            "options": ["Lorentz force", "Archimedes principle", "Pascal's law", "Hooke's law"],
            "answer": "A",
            "hint": "It is the force on moving charges in a magnetic field.",
            "explanation": "The force on a current-carrying conductor in a magnetic field is described by the Lorentz force."
        },
        {
            "question": "The unit of power is:",
            "options": ["Watt", "Joule", "Coulomb", "Ampere"],
            "answer": "A",
            "hint": "Power = work done per unit time.",
            "explanation": "Power is measured in watts, where 1 W = 1 J/s."
        },
        {
            "question": "When a body is charged by friction, the charge transfer is due to:",
            "options": ["Transfer of electrons", "Transfer of protons", "Transfer of neutrons", "Loss of energy"],
            "answer": "A",
            "hint": "Electrons are mobile in atoms and materials.",
            "explanation": "Charging by friction occurs due to transfer of electrons from one object to another."
        },
        {
            "question": "The direction of induced current is given by:",
            "options": ["Lenz's law", "Ohm's law", "Newton's law", "Gauss's law"],
            "answer": "A",
            "hint": "It states that induced current opposes the change in magnetic flux.",
            "explanation": "Lenz's law gives the direction of induced current, opposing the change causing it."
        },
    ]

    questions = physics_questions if (subject or "").lower() in {"physics", "phy"} else [
        {
            "question": "Which of the following is a correct statement about scientific study?",
            "options": ["It follows a systematic method", "It uses no observation", "It ignores evidence", "It is based only on guesses"],
            "answer": "A",
            "hint": "Science relies on observation and logic.",
            "explanation": "Scientific study uses evidence, observation, and reasoning."
        }
        for _ in range(1)
    ]

    selected = questions[:requested_count]
    if len(selected) < requested_count:
        already_selected = {item["question"] for item in selected}
        extra = [
            item
            for item in physics_questions
            if item["question"] not in already_selected
        ]
        selected.extend(extra[:requested_count - len(selected)])

    normalized_questions = []
    for idx, item in enumerate(selected, start=1):
        normalized_questions.append({
            "id": idx,
            "question": item["question"],
            "options": [
                {"id": letter, "text": option}
                for letter, option in zip("ABCD", item["options"])
            ],
            "answer": item["answer"],
            "hint": item["hint"],
            "explanation": item["explanation"],
        })

    return {
        "type": "quiz",
        "quiz": {
            "title": f"Class {class_level} {subject_text} Quiz",
            "questions": normalized_questions,
        }
    }


def top_up_quiz_questions(
    quiz_data: Dict[str, Any],
    requested_count: int,
    class_level: str,
    subject: Optional[str],
    chapter: Optional[str],
    user_message: str,
) -> Dict[str, Any]:
    """Pad a short quiz with unique fallback questions so the student
    always receives the number of questions they asked for."""

    questions = []

    if isinstance(quiz_data, dict) and isinstance(
        quiz_data.get("quiz"), dict
    ):
        questions = quiz_data["quiz"].get("questions") or []

    if len(questions) >= requested_count:
        return quiz_data

    seen = {
        str(q.get("question", "")).strip().casefold()
        for q in questions
        if isinstance(q, dict)
    }

    bank = generate_fallback_mcq_quiz(
        class_level,
        subject,
        chapter,
        user_message,
    )

    for extra in bank["quiz"]["questions"]:
        if len(questions) >= requested_count:
            break
        key = str(extra.get("question", "")).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        questions.append(extra)

    for index, question in enumerate(questions, start=1):
        if isinstance(question, dict):
            question["id"] = index

    if isinstance(quiz_data, dict):
        if not isinstance(quiz_data.get("quiz"), dict):
            quiz_data["quiz"] = {}
        quiz_data["quiz"]["questions"] = questions

    return quiz_data


def build_quiz_intro_message(
    actual_count: int,
    requested_count: int,
) -> str:
    """A short, polite and cheerful line shown with every quiz."""

    if actual_count < requested_count:
        return (
            f"Sure thing! 🌟 Here are {actual_count} MCQs for you. "
            f"I couldn't fit all {requested_count} in this time, "
            "but please try these and ask me for more afterwards! 💪😄"
        )

    return (
        f"Absolutely! Here are your {actual_count} MCQs 🎯 "
        "Take your time, read carefully, and best of luck! 🍀✨"
    )


def detect_subject(text: str) -> Optional[str]:

    text = normalize_text(text)

    if not text:
        return None


    subject_aliases = {
    "Physics" : ["Physics","physics","physic","phy","Class 12 Physics","Class XII Physics","CBSE Physics","NCERT Physics","Physics Numericals","Physics Concepts","Physics Formula","Physics Derivation","Physics Definitions","Physics Laws","Physics Principles","Physics Applications","Physics Diagrams","Physics Graphs","Physics SI Units","Physics Dimensions","Solved Examples","MCQ","Assertion Reason","Case Study","NCERT Questions","In-text Questions","Exercise Questions","Important Questions","Board Questions","Previous Year Questions","Electric Charges and Fields","Electric Charge","Coulomb's Law","Electric Field","Electric Field Lines","Electric Flux","Gauss's Law","Electric Dipole","Electric Dipole Moment","Electrostatic Potential and Capacitance","Electric Potential","Potential Difference","Equipotential Surface","Electrostatic Potential Energy","Capacitance","Capacitor","Parallel Plate Capacitor","Dielectric","Combination of Capacitors","Energy Stored in Capacitor","Current Electricity","Electric Current","Drift Velocity","Mobility","Ohm's Law","Resistance","Resistivity","Conductivity","EMF","Internal Resistance","Kirchhoff's Laws","Wheatstone Bridge","Meter Bridge","Potentiometer","Moving Charges and Magnetism","Magnetic Field","Lorentz Force","Biot-Savart Law","Ampere's Circuital Law","Magnetic Force","Cyclotron","Moving Coil Galvanometer","Magnetic Dipole","Magnetism and Matter","Bar Magnet","Magnetic Gauss Law","Magnetisation","Magnetic Intensity","Magnetic Permeability","Diamagnetism","Paramagnetism","Ferromagnetism","Magnetic Materials","Electromagnetic Induction","Magnetic Flux","Faraday's Law","Lenz's Law","Motional EMF","Self Induction","Mutual Induction","Inductor","Eddy Currents","Alternating Current","AC Voltage","AC Circuit","RMS Value","Reactance","Inductive Reactance","Capacitive Reactance","Impedance","LCR Circuit","Resonance","Power in AC Circuit","Transformer","Electromagnetic Waves","Displacement Current","Electromagnetic Spectrum","Radio Waves","Microwaves","Infrared Rays","Visible Light","Ultraviolet Rays","X-Rays","Gamma Rays","Ray Optics and Optical Instruments","Ray Optics","Reflection","Refraction","Spherical Mirrors","Mirror Formula","Lens","Lens Formula","Lens Maker's Formula","Power of Lens","Prism","Total Internal Reflection","Optical Fibre","Human Eye","Microscope","Telescope","Wave Optics","Huygens Principle","Wavefront","Interference","Young's Double Slit Experiment","Coherent Sources","Fringe Width","Diffraction","Polarisation","Dual Nature of Radiation and Matter","Photoelectric Effect","Photoelectric Equation","Work Function","Threshold Frequency","Stopping Potential","de Broglie Hypothesis","de Broglie Wavelength","Matter Waves","Atoms","Rutherford Model","Bohr Model","Bohr's Postulates","Hydrogen Atom","Energy Levels","Atomic Spectrum","Hydrogen Spectrum","Nuclei","Atomic Nucleus","Nuclear Size","Nuclear Mass","Mass Defect","Binding Energy","Nuclear Force","Radioactivity","Alpha Decay","Beta Decay","Gamma Decay","Nuclear Fission","Nuclear Fusion","Semiconductor Electronics","Semiconductor","Intrinsic Semiconductor","Extrinsic Semiconductor","P-Type Semiconductor","N-Type Semiconductor","PN Junction","Semiconductor Diode","Rectifier","Zener Diode","LED","Photodiode","Solar Cell","Logic Gates","AND Gate","OR Gate","NOT Gate","NAND Gate","NOR Gate"],
    "Chemistry": ["Chemistry","che","chem","chemesty","Class 12 Chemistry","Class XII Chemistry","CBSE Chemistry","NCERT Chemistry","Chemistry Numericals","Chemistry Concepts","Chemistry Formula","Chemistry Reactions","Chemistry Equations","Chemistry Definitions","Chemistry Mechanisms","Chemistry Properties","Chemistry Preparation","Chemistry Applications","Chemistry Diagrams","Solved Examples","MCQ","Assertion Reason","Case Study","NCERT Questions","In-text Questions","Exercise Questions","Important Questions","Board Questions","Previous Year Questions", "Solutions","Types of Solutions","Concentration","Molarity","Molality","Mole Fraction","Mass Percentage","Henry's Law","Raoult's Law","Colligative Properties","Osmosis","Osmotic Pressure","Elevation in Boiling Point","Depression in Freezing Point","Abnormal Molar Mass","Van't Hoff Factor","Electrochemistry","Electrochemical Cell","Galvanic Cell","Electrolytic Cell","Electrode Potential","Standard Electrode Potential","Nernst Equation","EMF","Cell Potential","Conductance","Conductivity","Molar Conductivity","Kohlrausch's Law","Electrolysis","Faraday's Laws","Batteries","Fuel Cell","Corrosion","Chemical Kinetics","Rate of Reaction","Rate Law","Order of Reaction","Molecularity","Integrated Rate Equation","Half Life","Zero Order Reaction","First Order Reaction","Arrhenius Equation","Activation Energy","Rate Constant","Temperature Dependence","d and f Block Elements","Transition Elements","d Block Elements","f Block Elements","Electronic Configuration","Oxidation States","Coordination Compounds","Magnetic Properties","Catalytic Properties","Alloys","Lanthanides","Actinides","Lanthanide Contraction","Coordination Compounds","Coordination Entity","Ligand","Coordination Number","Coordination Sphere","Werner's Theory","IUPAC Nomenclature","Isomerism","Geometrical Isomerism","Optical Isomerism","Crystal Field Theory","Valence Bond Theory","Crystal Field Splitting","Magnetic Behaviour","Haloalkanes and Haloarenes","Haloalkanes","Haloarenes","Alkyl Halides","Aryl Halides","C-X Bond","Nucleophilic Substitution","SN1 Reaction","SN2 Reaction","Elimination Reaction","Wurtz Reaction","Finkelstein Reaction","Swarts Reaction","Grignard Reagent","Alcohols Phenols and Ethers","Alcohols","Phenols","Ethers","Preparation of Alcohols","Preparation of Phenols","Preparation of Ethers","Dehydration","Oxidation of Alcohols","Acidity of Phenols","Williamson Ether Synthesis","Kolbe Reaction","Reimer-Tiemann Reaction","Aldehydes Ketones and Carboxylic Acids","Aldehydes","Ketones","Carboxylic Acids","Carbonyl Compounds","Nucleophilic Addition","Aldol Condensation","Cannizzaro Reaction","Clemmensen Reduction","Wolff-Kishner Reduction","Tollens Test","Fehling Test","Iodoform Test","Hell-Volhard-Zelinsky Reaction","Amines","Amines Class 12","Primary Amine","Secondary Amine","Tertiary Amine","Basicity of Amines","Preparation of Amines","Hoffmann Bromamide Reaction","Gabriel Phthalimide Synthesis","Diazotisation","Diazonium Salts","Carbylamine Reaction","Hinsberg Test","Sandmeyer Reaction","Biomolecules","Carbohydrates","Glucose","Fructose","Sucrose","Starch","Cellulose","Proteins","Amino Acids","Peptide Bond","Protein Structure","Enzymes","Vitamins","Nucleic Acids","DNA","RNA","Polymers","Polymerisation","Addition Polymerisation","Condensation Polymerisation","Copolymerisation","Thermoplastic","Thermosetting Polymer","Elastomers","Nylon","Bakelite","Teflon","PVC","Natural Rubber","Synthetic Rubber","Chemistry in Everyday Life","Drugs","Medicines","Analgesics","Antibiotics","Antiseptics","Disinfectants","Antacids","Antihistamines","Tranquilizers","Food Preservatives","Artificial Sweeteners","Soaps","Detergents"],
    "Computer-Science" : ["Computer Science","cs","c s","comp","computer","Class 12 Computer Science","Class XII Computer Science","CBSE Computer Science","NCERT Computer Science","Computer Science Python","Programming","Python","Python Programming","Variables","Data Types","Strings","Lists","Tuples","Dictionaries","Operators","Expressions","Type Conversion","Conditional Statements","If Statement","If Else","Nested If","Loops","For Loop","While Loop","Functions","User Defined Functions","Built-in Functions","Function Arguments","Parameters","Return Statement","Scope of Variables","Local Variable","Global Variable","Modules",  "Data Structures","Stack","Queue","List Operations","Push","Pop","Enqueue","Dequeue","LIFO","FIFO",  "File Handling","Text File","Binary File","CSV File","File Modes","Open Function","File Pointer","Tell","Seek","Exception Handling","Errors","Syntax Error","Runtime Error","Logical Error","Database","DBMS","Database Management System","Relational Database","RDBMS","Table","Record","Field","Tuple","Attribute","Primary Key","Candidate Key","Foreign Key","SQL","MySQL","SQL Commands","DDL","DML","DQL","CREATE","ALTER","DROP","INSERT","UPDATE","DELETE","DISTINCT","ORDER BY","GROUP BY","BETWEEN","IS NULL","SQL Functions","Aggregate Functions","AVG","MAX","MIN","String Functions","Math Functions","Date Functions","SQL Joins","JOIN","Equi Join","Natural Join","Cartesian Product","Computer Networks","Networking","Network","LAN","MAN","WAN","PAN","Network Topology","Bus Topology","Star Topology","Ring Topology","Mesh Topology","Network Devices","Hub","Switch","Router","Modem","Repeater","Gateway","IP Address","IPv4","IPv6","MAC Address","Protocol","Network Protocol","Bandwidth","Data Transfer","Internet","WWW","World Wide Web","URL","Web Browser","Web Server","DNS","Domain Name","HTTP","HTTPS","FTP","SMTP","POP3","TCP","UDP","IP","TELNET","Cyber Security","Cyber Safety","Cyber Crime","Phishing","Malware","Virus","Worm","Trojan","Firewall","Data Security","Privacy","Digital Footprint","Cyber Ethics","Syntax","Program","Python Program","Error","Debugging","MCQ","Assertion Reason","Case Study","NCERT Questions","In-text Questions","Exercise Questions","Important Questions","Board Questions","Previous Year Questions"],
    "Mathematics" : ["Mathematics","mathematics","maths","math","mathematic","Maths","Class 12 Mathematics","Class XII Mathematics","CBSE Mathematics","NCERT Mathematics","Class 12 Maths","Maths Numericals","Mathematical Concepts","Maths Formula","Maths Derivation","Solved Examples","MCQ","Assertion Reason","Case Study","NCERT Questions","In-text Questions","Exercise Questions","Important Questions","Board Questions","Previous Year Questions",  "Relations and Functions","Relations","Types of Relations","Equivalence Relation","Functions","Types of Functions","One-One Function","Many-One Function","Onto Function","Into Function","Composition of Functions","Invertible Function","Inverse Function","Binary Operations", "Inverse Trigonometric Functions","Inverse Trigonometric Function","Principal Values","Domain","Range","Inverse Sine","Inverse Cosine","Inverse Tangent","Inverse Trigonometric Identities",  "Matrices","Matrix","Types of Matrices","Row Matrix","Column Matrix","Square Matrix","Diagonal Matrix","Scalar Matrix","Identity Matrix","Zero Matrix","Symmetric Matrix","Skew Symmetric Matrix","Matrix Operations","Matrix Addition","Matrix Multiplication","Transpose of Matrix","Inverse of Matrix","Determinants","Determinant","Properties of Determinants","Area of Triangle","Minors","Cofactors","Adjoint","Inverse Using Adjoint","Linear Equations","System of Linear Equations","Cramer's Rule","Continuity and Differentiability","Continuity","Differentiability","Derivative","Derivatives","Differentiation","Chain Rule","Product Rule","Quotient Rule","Implicit Differentiation","Logarithmic Differentiation","Parametric Differentiation","Second Order Derivative","Mean Value Theorem","Rolle's Theorem","Lagrange's Mean Value Theorem","Applications of Derivatives","Rate of Change","Increasing Functions","Decreasing Functions","Monotonicity","Maxima","Minima","Local Maximum","Local Minimum","Critical Points","Optimization","Integrals","Integration","Indefinite Integral","Definite Integral","Integration by Substitution","Integration by Parts","Partial Fractions","Standard Integrals","Properties of Definite Integrals","Applications of Integrals","Area Under Curve","Area Between Curves","Area Bounded by Curves","Differential Equations","Differential Equation","Order of Differential Equation","Degree of Differential Equation","General Solution","Particular Solution","Variable Separable Method","Homogeneous Differential Equation","Linear Differential Equation","Vector Algebra","Vectors","Vector","Magnitude of Vector","Unit Vector","Direction Cosines","Direction Ratios","Position Vector","Section Formula","Dot Product","Scalar Product","Cross Product","Vector Product","Projection of Vector","Scalar Triple Product","Three Dimensional Geometry","3D Geometry","Direction Cosines","Direction Ratios","Line in 3D","Equation of Line","Cartesian Equation","Vector Equation","Angle Between Lines","Distance Between Lines","Shortest Distance","Plane","Equation of Plane","Angle Between Planes","Distance Between Point and Plane","Linear Programming","Linear Programming Problem","LPP","Objective Function","Constraints","Feasible Region","Feasible Solution","Optimal Solution","Graphical Method","Probability","Conditional Probability","Independent Events","Multiplication Theorem","Bayes Theorem","Random Variable","Probability Distribution","Mean of Random Variable","Variance","Binomial Distribution","Bernoulli Trials","Formula","Theorem","Proof","Identity","Equation","Inequality","Graph","Domain","Range","Limit","Calculation","Solution","Step by Step Solution","Solve","Simplify","Evaluate","Find","Prove","Derive",],
    "Biology" : ["Biology","Class 12 Biology","Class XII Biology","CBSE Biology","NCERT Biology","Class 12 Bio","Biology Concepts","Biology Definitions","Biology Diagrams","Biology Processes","Biology Examples","Biology Questions","MCQ bio","Assertion Reason of bio","Case Study of bio ","NCERT Questions of bio","In-text Questions of bio ","Exercise Questions of bio","Important Questions of bio","Board Questions of bio","Previous Year Questions of bio " , "Reproduction","Sexual Reproduction","Asexual Reproduction","Reproduction in Organisms","Flowering Plants","Sexual Reproduction in Flowering Plants","Human Reproduction","Reproductive System","Male Reproductive System","Female Reproductive System","Gametogenesis","Spermatogenesis","Oogenesis","Menstrual Cycle","Fertilisation","Implantation","Pregnancy","Embryonic Development","Reproductive Health","Contraception","Infertility",  "Genetics","Heredity","Variation","Mendel's Laws","Mendelian Inheritance","Monohybrid Cross","Dihybrid Cross","Test Cross","Back Cross","Incomplete Dominance","Codominance","Multiple Alleles","Blood Groups","Sex Determination","Sex Linked Inheritance","Genetic Disorders","Pedigree Analysis","Molecular Basis of Inheritance","DNA","RNA","Replication","Transcription","Translation","Genetic Code","Gene Expression","Lac Operon","Mutation","DNA Fingerprinting","Evolution","Origin of Life","Darwinism","Natural Selection","Evidence of Evolution","Human Evolution","Human Health and Disease","Health","Disease","Pathogens","Bacteria","Viruses","Protozoans","Fungi","Parasites","Common Diseases","Immunity","Innate Immunity","Acquired Immunity","Antibodies","Vaccination","AIDS","Cancer","Allergy","Drug and Alcohol Abuse","Microbes in Human Welfare","Microorganisms","Fermentation","Industrial Production","Antibiotics","Sewage Treatment","Biogas","Biofertilisers","Biocontrol Agents","Biotechnology","Biotechnology Principles and Processes","Recombinant DNA Technology","Genetic Engineering","DNA Technology","Restriction Enzymes","DNA Ligase","Vector","Plasmid","Cloning","PCR","Gel Electrophoresis","Transformation","Bioreactor","Downstream Processing","Biotechnology and its Applications","Genetically Modified Organisms","GM Crops","Insulin Production","Gene Therapy","Molecular Diagnosis","Transgenic Animals","Biosafety","Biopiracy","Patents","Ecology","Organisms and Populations","Population","Population Attributes","Population Growth","Exponential Growth","Logistic Growth","Population Interactions","Predation","Competition","Parasitism","Mutualism","Commensalism","Ecosystem","Ecosystem Structure","Food Chain","Food Web","Trophic Levels","Ecological Pyramids","Energy Flow","Ecological Succession","Nutrient Cycling","Carbon Cycle","Phosphorus Cycle","Biodiversity","Biodiversity and Conservation","Species Diversity","Genetic Diversity","Ecosystem Diversity","Biodiversity Hotspots","Endangered Species","Extinction","Conservation","In Situ Conservation","Ex Situ Conservation","National Parks","Wildlife Sanctuaries","Biosphere Reserves","Environmental Issues","Air Pollution","Water Pollution","Soil Pollution","Solid Waste","Global Warming","Greenhouse Effect","Ozone Depletion","Acid Rain","Deforestation","Sustainable Development"],
    
}

    scores = {
        subject: 0
        for subject in subject_aliases
    }

    for subject, aliases in subject_aliases.items():

        for alias in aliases:

            alias = normalize_text(alias)

            if not alias:
                continue

            if re.search(
                rf"\b{re.escape(alias)}\b",
                text
            ):
                scores[subject] += 10

    # Existing chapter/topic dictionary also contributes
    for subject, chapters in CHAPTER_KEYWORDS.items():

        for chapter, keywords in chapters.items():

            chapter_name = normalize_text(chapter)

            if chapter_name and re.search(
                rf"\b{re.escape(chapter_name)}\b",
                text
            ):
                scores[subject] += 12

            for keyword in keywords:

                keyword = normalize_text(keyword)

                if keyword and re.search(
                    rf"\b{re.escape(keyword)}\b",
                    text
                ):
                    scores[subject] += min(
                        5,
                        max(1, len(keyword.split()))
                    )

    best_subject = max(
        scores,
        key=scores.get
    )

    if scores[best_subject] <= 0:
        return None

    return best_subject


def detect_chapter(
    text: str,
    subject: Optional[str]
) -> Optional[str]:

    if not subject:
        return None

    if subject == "Mathematics":
        chapter_subject = "Mathematics"
    else:
        chapter_subject = subject

    if chapter_subject not in CHAPTER_KEYWORDS:
        return None

    text = normalize_text(text)

    if not text:
        return None

    scores = {}

    for chapter, keywords in CHAPTER_KEYWORDS [chapter_subject].items():

        score = 0

        chapter_normalized = normalize_text(chapter)

        if chapter_normalized in text:
            score += 15

        for keyword in keywords:

            keyword = normalize_text(keyword)

            if not keyword:
                continue

            if keyword in text:
                words = len(keyword.split())
                if words >= 4:
                    score += 8
                elif words == 3:
                    score += 6
                elif words == 2:
                    score += 4
                else:
                    score += 1

        scores[chapter] = score

    best_chapter = max(
        scores,
        key=scores.get
    )

    if scores[best_chapter] == 0:
        return None

    return best_chapter

def get_recent_history_text(
    history: List[HistoryMessage]
) -> str:

    if not history:
        return ""

    recent = history[-MAX_HISTORY_MESSAGES:]

    parts = []

    for item in recent:

        if item.role not in {
            "user",
            "assistant"
        }:
            continue

        content = item.content.strip()

        if content:
            parts.append(content)

    return "\n".join(parts)


def detect_from_conversation(
    current_message: str,
    history: List[HistoryMessage]
):

    history_text = get_recent_history_text(history)

    combined_text = (
        history_text
        + "\n"
        + current_message
    )

    subject = detect_subject(
        combined_text
    )

    chapter = detect_chapter(
        combined_text,
        subject
    )

    return subject, chapter


def resolve_class(
    request: ChatRequest
) -> str:

    if request.class_level:
        value = str(
            request.class_level
        ).strip()

        match = re.search(
            r"\d+",
            value
        )

        if match:
            return match.group(0)

    text = normalize_text(
        request.message
    )

    match = re.search(
        r"class\s*(9|10|11|12)\b",
        text
    )

    if match:
        return match.group(1)

    match = re.search(
        r"class\s*(ix|x|xi|xii)\b",
        text
    )

    if match:
        roman = match.group(1)

        roman_map = {
            "ix": "9",
            "x": "10",
            "xi": "11",
            "xii": "12"
        }

        return roman_map[roman]

    return "12"


def should_use_rag(
    text: str,
    subject: Optional[str],
    chapter: Optional[str]
) -> bool:

    text = normalize_text(text)

    if not text:
        return False

    # Casual conversation
    casual = {
        "hi",
        "hii",
        "hello",
        "hey",
        "hlo",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "bye",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "who are you",
        "what are you",
        "how are you",
        "tum kaun ho",
        "aap kaun ho",
        "kaise ho"
    }

    if text in casual:
        return False

    # Current-news/general web questions
    news_terms = [
        "today news",
        "aaj ki news",
        "aaj ki khabar",
        "latest news",
        "breaking news",
        "current news",
        "today's news"
    ]

    if any(term in text for term in news_terms):
        return False

    # Explicit NCERT request
    ncert_terms = [
        "ncert",
        "textbook",
        "ncert ke according",
        "ncert according",
        "book ke according",
        "ncert me",
        "ncert mein",
        "ncert definition",
        "ncert example",
        "ncert question",
        "exercise question",
        "intext question"
    ]

    if any(
        term in text
        for term in ncert_terms
    ):
        return True

    # Specific chapter detected
    if chapter:
        return True

    # Subject + academic question
    academic_terms = [
        "define",
        "definition",
        "explain",
        "samjhao",
        "samjha",
        "meaning",
        "formula",
        "equation",
        "derive",
        "derivation",
        "numerical",
        "calculate",
        "solve",
        "difference",
        "compare",
        "prove",
        "proof",
        "law",
        "theorem",
        "principle",
        "concept",
        "example",
        "properties",
        "reaction",
        "mechanism",
        "graph",
        "diagram",
        "units",
        "dimension",
        "mcq",
        "questions",
        "question"
    ]

    if subject and any(
        term in text
        for term in academic_terms
    ):
        return True

    # Long academic question
    if subject and len(text.split()) >= 6:
        return True

    return False


def is_quiz_request(text: str) -> bool:
    text = normalize_text(text)

    # Definition/explanation questions about MCQs or quizzes are
    # NOT requests to generate a quiz.
    definition_phrases = [
        "what is mcq",
        "what is an mcq",
        "what is mcqs",
        "what are mcqs",
        "define mcq",
        "meaning of mcq",
        "explain mcq",
        "about mcq",
        "about mcqs",
        "mcq kya hai",
        "mcq kya hota hai",
        "what is quiz",
        "what is a quiz",
        "quiz kya hai",
        "explain quiz",
        "what is multiple choice",
        "difference between mcq",
    ]

    if any(phrase in text for phrase in definition_phrases):
        return False

    quiz_terms = [
        "mcq",
        "mcqs",
        "multiple choice",
        "multiple choice questions",
        "quiz",
        "test me",
        "practice questions",
        "objective questions",
        "objective question"
    ]

    return any(term in text for term in quiz_terms)

@lru_cache(maxsize=100)
def generate_query_embedding_cached(
    text: str
):

    result = gemini_embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "output_dimensionality":
                EMBEDDING_DIMENSION
        }
    )

    return tuple(
        result.embeddings[0].values
    )


def generate_query_embedding(
    text: str
):

    clean_text = " ".join(
        text.lower().strip().split()
    )

    return list(
        generate_query_embedding_cached(
            clean_text
        )
    )


def search_ncert(
    query: str,
    class_level: str,
    subject: Optional[str] = None,
    chapter: Optional[str] = None
):

    try:

        embedding_start = time.perf_counter()

        query_embedding = (
            generate_query_embedding(
                query
            )
        )

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )

        supabase_start = (
            time.perf_counter()
        )

        response = supabase.rpc(
            "match_ncert_documents",
            {
                "query_embedding":
                    query_embedding,
                "match_threshold":
                    RAG_MATCH_THRESHOLD,
                "match_count":
                    RAG_MATCH_COUNT,
                "filter_class_level":
                    class_level,
                "filter_subject":
                    subject,
                "filter_chapter":
                    chapter
            }
        ).execute()

        supabase_time = (
            time.perf_counter()
            - supabase_start
        )

        print(
            f"Embedding: {embedding_time:.2f}s | "
            f"Supabase: {supabase_time:.2f}s"
        )

        return response.data or []

    except Exception as e:

        print(
            "RAG ERROR:",
            repr(e)
        )

        return []


def build_ncert_context(
    results
):

    if not results:
        return ""

    parts = []
    total_chars = 0

    for index, item in enumerate(
        results,
        start=1
    ):

        content = item.get(
            "content",
            ""
        ).strip()

        if not content:
            continue

        content = content [:MAX_CHUNK_CHARS]

        source_name = item.get(
            "source_name",
            "NCERT"
        )

        source_page = item.get(
            "source_page"
        )

        chapter = item.get(
            "chapter"
        )

        topic = item.get(
            "topic"
        )

        header = (
            f"[NCERT SOURCE {index}]"
        )

        if chapter:
            header += (
                f"\nChapter: {chapter}"
            )

        if topic:
            header += (
                f"\nTopic: {topic}"
            )

        if source_name:
            header += (
                f"\nBook: {source_name}"
            )

        if source_page:
            header += (
                f"\nPage: {source_page}"
            )

        part = (
            f"{header}\n"
            f"Content:\n{content}"
        )

        remaining = (
            MAX_NCERT_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 200:
            break

        if len(part) > remaining:
            part = part[:remaining]

        parts.append(part)

        total_chars += len(part)

    return "\n\n---\n\n".join(
        parts
    )


def build_system_prompt(
    class_level: str,
    subject: Optional[str],
    chapter: Optional[str],
    ncert_context: str
):

    subject_text = (
        subject
        or "Not explicitly detected"
    )

    chapter_text = (
        chapter
        or "Not explicitly detected"
    )

    context_text = (
        ncert_context
        if ncert_context
        else "No matching NCERT context was retrieved."
    )

    return f"""
You are Study Notes Pro AI.

You are a helpful CBSE Class {class_level} educational assistant.

ACTIVE CLASS:
Class {class_level}

ACTIVE SUBJECT:
{subject_text}

ACTIVE CHAPTER:
{chapter_text}

IMPORTANT FALLBACK BEHAVIOR:

1. Class {class_level} is the default class.

2. NEVER ask the student for subject or chapter merely because
   subject/chapter detection failed.

3. First try to understand the student's question yourself.

4. If the question is understandable from general knowledge
   or your internal knowledge, answer it directly.

5. If subject is known but chapter is unknown, answer using
   the subject context without asking for the chapter.

6. If chapter is known, prefer the supplied NCERT context.

7. If NCERT context is unavailable, do NOT refuse the question.
   Answer using your own knowledge while staying aligned with
   Class {class_level} CBSE level.

8. Only ask a clarification question when the question is
   genuinely impossible to answer without additional information.

9. If previous conversation clearly establishes the subject,
   chapter or topic, use that context.

10. Do not repeatedly ask the student to provide information
    that can reasonably be inferred.

ACADEMIC RULES:

- Default to Class {class_level}.
- Prefer CBSE/NCERT terminology.
- Do not silently switch to another class.
- Give direct answers first.
- Explain concepts clearly.
- For numericals, show steps.
- For formulas, explain symbols.
- Keep equations readable.
- Do not invent NCERT citations or page numbers.
- If supplied NCERT context exists, prefer it.

LANGUAGE RULE:

- DEFAULT REPLY LANGUAGE IS ENGLISH.
- If the user writes in English, always respond in English.
- If the user writes in Hinglish (Hindi written in English/Latin
  letters), respond in Hinglish.
- If the user writes in Hindi (Devanagari script), respond in Hinglish
  unless the user explicitly asks for Devanagari Hindi.
- Allowed default reply languages: English and Hinglish only.
- Do not force Hindi when the user is using English.
- Do not force English when the user is using Hinglish.

FRIENDLY REPLY RULE:

- Always reply in a warm, polite and encouraging tone.
- Use friendly words such as "Sure!", "Absolutely!", "Here you go",
  "Happy to help!" and "Great question!" where they fit naturally.
- Include 1 to 3 relevant emojis in every reply (for example 📚, ✨,
  🎯, 💡, 🍀, 😊, 💪) so the chat feels lively — but do not overdo it.
- Encourage the student, stay patient, and never sound rude or robotic.
- If the student makes a mistake, respond kindly and help them learn.

MCQ RULES:

If the student asks for multiple MCQs:

1. Generate exactly the requested number.
2. Each question has exactly four options.
3. Options must be A, B, C and D.
4. Do not reveal the answer after each question.
5. Put the complete answer key at the end.
6. Answer key count must equal MCQ count.
7. If the student says "10 MCQ", generate exactly 10.
8. If subject/chapter is reasonably inferable from the conversation,
   use it without asking for clarification.

FORMAT:

###  Question 01

**Question text**

**A)** Option A
**B)** Option B
**C)** Option C
**D)** Option D

Then continue sequentially.

At the end:

# 🏆 ANSWER KEY

01. A
02. B
...

NCERT CONTEXT:

{context_text}
"""




def build_quiz_prompt(
    class_level: str,
    subject: Optional[str],
    chapter: Optional[str],
    ncert_context: str,
    user_message: str
) -> str:

    subject_text = subject or "Not explicitly detected"
    chapter_text = chapter or "Not explicitly detected"

    context_text = (
        ncert_context
        if ncert_context
        else "No matching NCERT context was retrieved."
    )

    requested_count = extract_requested_question_count(user_message)

    return f"""
You are the quiz generator for Study Notes Pro.

Generate a CBSE Class {class_level} quiz.

CLASS:
{class_level}

SUBJECT:
{subject_text}

CHAPTER:
{chapter_text}

STUDENT REQUEST:
{user_message}

NUMBER OF QUESTIONS:
{requested_count}

IMPORTANT:

- The questions array MUST contain exactly {requested_count} questions.
- Do not stop early, do not skip questions, do not truncate the JSON.
- Keep every hint and explanation under 15 words so the JSON always fits.
- Every question must have exactly 4 options.
- Options must be A, B, C and D.
- Only one option can be correct.
- Questions should be suitable for CBSE Class {class_level}.
- Prefer NCERT content when NCERT context is provided.
- Do not invent facts.
- Do not reveal the correct answer inside the question.
- Every question must contain a short hint.
- Every question must contain a short explanation.
- Explanations should be shown only after the student answers.
- Questions should test understanding, not only memorization.
- Write questions in English by default, or in Hinglish if the
  student is writing in Hinglish.
- Do not use Devanagari Hindi unless the student explicitly
  asks for it.

RETURN ONLY VALID JSON.
IMPORTANT ACADEMIC ACCURACY RULES:

1. Questions MUST match the detected Class, Subject, Chapter and Topic.
2. Do not mix chapters unless the user explicitly asks for a mixed quiz.
3. Use NCERT/RAG context as the primary source whenever available.
4. Never invent an answer.
5. Verify every numerical answer before returning the quiz.
6. Verify every conceptual answer against standard NCERT physics.
7. Every MCQ must have exactly ONE unambiguously correct answer.
8. Never create a question where two options can both be correct.
9. For symmetry-based questions, do not ask for a single charge if two or more charges have equal force unless the question explicitly allows multiple answers.
10. For Physics calculations, check units and numerical values carefully.

PHYSICS FACT CHECK:
- When glass is rubbed with silk:
  glass becomes POSITIVELY charged.
  silk becomes NEGATIVELY charged.
- Do not reverse this fact.
MATHEMATICAL NOTATION RULES:

- Use LaTeX for mathematical expressions.
- Inline math MUST use \\( ... \\)
- Display math MUST use \\[ ... \\]
- Never write raw LaTeX commands without delimiters.
- Never write r2 when you mean r².
- Write r^2.
- Never write pm2 when you mean m².


The JSON must have exactly this structure:

{{
  "type": "quiz",
  "quiz": {{
    "title": "Class {class_level} {subject_text} Quiz",
    "questions": [
      {{
        "id": 1,
        "question": "Question text",
        "options": [
          {{
            "id": "A",
            "text": "Option A"
          }},
          {{
            "id": "B",
            "text": "Option B"
          }},
          {{
            "id": "C",
            "text": "Option C"
          }},
          {{
            "id": "D",
            "text": "Option D"
          }}
        ],
        "answer": "B",
        "hint": "Short hint",
        "explanation": "Short explanation"
      }}
    ]
  }}
}}

The answer field MUST contain only:
A
B
C
or D

NCERT CONTEXT:

{context_text}
"""
def _strip_quiz_fences(text: str) -> str:
    # Remove ```json ... ``` fences anywhere in the text
    text = re.sub(
        r"```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.replace("```", "").strip()


def _extract_quiz_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    cleaned = _strip_quiz_fences(text.strip())
    if cleaned:
        candidates.append(cleaned)
    # Find balanced {...} substring containing '"type"' and '"quiz"'
    # so leading/trailing prose from the LLM does not break parsing.
    start_markers = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_markers:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start:idx + 1]
                        if (
                            '"quiz"' in snippet
                            or '"questions"' in snippet
                        ):
                            candidates.append(_strip_quiz_fences(snippet))
                        break
        if len(candidates) >= 6:
            break
    # Deduplicate while keeping order
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return unique


def _lenient_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Repair common LLM mistakes: trailing commas, single quotes edge case
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _normalize_quiz_options(
    raw_options: Any
) -> List[Dict[str, str]]:
    """Accept options as dicts, plain strings, or an {id: text} map."""

    entries = []

    if isinstance(raw_options, dict):
        for key, value in raw_options.items():
            if isinstance(value, dict):
                text = str(
                    value.get("text")
                    or value.get("label")
                    or value.get("content")
                    or ""
                ).strip()
                option_id = str(value.get("id") or key)
            else:
                text = str(value).strip()
                option_id = str(key)
            entries.append((option_id, text))

    elif isinstance(raw_options, list):
        for item in raw_options:
            if isinstance(item, dict):
                text = str(
                    item.get("text")
                    or item.get("label")
                    or item.get("content")
                    or ""
                ).strip()
                option_id = str(item.get("id") or "")
            elif isinstance(item, str):
                text = item.strip()
                option_id = ""
            else:
                continue
            entries.append((option_id, text))

    cleaned: List[Dict[str, str]] = []

    for option_id, text in entries:

        if not text or len(cleaned) >= 4:
            continue

        used_ids = {
            option["id"]
            for option in cleaned
        }

        option_id = option_id.strip().upper()

        if (
            option_id not in {"A", "B", "C", "D"}
            or option_id in used_ids
        ):
            option_id = next(
                (
                    letter
                    for letter in "ABCD"
                    if letter not in used_ids
                ),
                ""
            )

        if not option_id:
            continue

        cleaned.append({
            "id": option_id,
            "text": text
        })

    return cleaned


def _resolve_quiz_answer(
    raw_answer: Any,
    options: List[Dict[str, str]]
) -> str:
    """Map a letter, number, or full-text answer onto an option id."""

    answer = str(
        raw_answer or ""
    ).strip()

    if not answer or not options:
        return ""

    letter_match = re.fullmatch(
        r"\W*([A-Da-d])\W*",
        answer
    )

    if letter_match:
        letter = letter_match.group(1).upper()
        if any(
            option["id"] == letter
            for option in options
        ):
            return letter

    if re.fullmatch(r"[1-4]", answer):
        index = int(answer) - 1
        if 0 <= index < len(options):
            return options[index]["id"]

    target = answer.casefold()

    for option in options:
        if (
            option["text"].strip().casefold()
            == target
        ):
            return option["id"]

    if len(target) >= 4:
        for option in options:
            text = (
                option["text"]
                .strip()
                .casefold()
            )
            if len(text) >= 3 and (
                text in target
                or target in text
            ):
                return option["id"]

    return ""


def _looks_like_quiz_json(text: Optional[str]) -> bool:
    """True when a reply is a raw JSON blob, not normal prose."""

    if not text:
        return False

    stripped = _strip_quiz_fences(text).lstrip()

    return stripped.startswith("{") and (
        '"quiz"' in stripped
        or '"questions"' in stripped
    )


def parse_quiz_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # The model may return the whole payload as one JSON-encoded string.
    stripped_head = _strip_quiz_fences(text).lstrip()

    if stripped_head.startswith('"'):
        try:
            unwrapped = json.loads(stripped_head)
        except json.JSONDecodeError:
            unwrapped = None

        if isinstance(unwrapped, str):
            text = unwrapped

    for candidate in _extract_quiz_json_candidates(text):
        data = _lenient_json_loads(candidate)
        if not data:
            continue

        # The payload may come wrapped in {"type": "quiz", "quiz": {...}}
        # or as a bare object with questions, sometimes with no type.
        quiz: Optional[Dict[str, Any]] = None

        if isinstance(data.get("quiz"), dict):
            quiz = data["quiz"]
        elif isinstance(data.get("quiz"), list):
            quiz = {"questions": data["quiz"]}
        elif isinstance(data.get("questions"), list):
            quiz = data

        if not isinstance(quiz, dict):
            continue

        questions = quiz.get("questions")

        if not isinstance(questions, list):
            continue

        cleaned_questions: List[Dict[str, Any]] = []

        for index, question in enumerate(
            questions,
            start=1
        ):

            if not isinstance(question, dict):
                continue

            question_text = str(
                question.get("question")
                or question.get("text")
                or ""
            ).strip()

            if not question_text:
                continue

            raw_options = question.get("options")

            if raw_options is None:
                raw_options = question.get("choices")

            cleaned_options = _normalize_quiz_options(
                raw_options
            )

            if len(cleaned_options) < 2:
                continue

            answer = _resolve_quiz_answer(
                question.get("answer")
                or question.get("correct")
                or question.get("correctAnswer"),
                cleaned_options
            )

            if not answer:
                continue

            cleaned_questions.append({
                "id": index,
                "question": question_text,
                "options": cleaned_options,
                "answer": answer,
                "hint": str(
                    question.get("hint")
                    or ""
                ).strip(),
                "explanation": str(
                    question.get("explanation")
                    or ""
                ).strip()
            })

        if not cleaned_questions:
            continue

        return {
            "type": "quiz",
            "quiz": {
                "title": str(
                    quiz.get("title")
                    or "Study Notes Pro Quiz"
                ).strip(),
                "questions": cleaned_questions
            }
        }

    return None

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Study Notes Pro AI"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/api/chat")
def chat(
    request: ChatRequest
):

    total_start = time.perf_counter()

    try:

        current_message = (
            request.message.strip()
        )

        if (
            not current_message
            and not request.attachments
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Message or attachment "
                    "is required."
                )
            )

        routing_start = (
            time.perf_counter()
        )

        active_class = resolve_class(
            request
        )

        # -------------------------------------------------
        # SMART CONTEXT DETECTION
        # -------------------------------------------------

        subject, chapter = detect_from_conversation(
           current_message,
          request.history
      )

     # Explicit request values get priority
        if request.subject:
            subject = request.subject

        if request.chapter:
         chapter = request.chapter

        use_rag = should_use_rag(
         current_message,
         subject,
         chapter
        )


        routing_time = (
            time.perf_counter()
            - routing_start
        )

        print(
            "\n========== ROUTING =========="
        )

        print(
            "Class   :",
            active_class
        )

        print(
            "Subject :",
            subject
        )

        print(
            "Chapter :",
            chapter
        )

        print(
            "RAG     :",
            use_rag
        )

        print(
            "Question:",
            current_message
        )

        print(
            f"Routing : {routing_time:.3f}s"
        )

        print(
            "============================="
        )

        rag_results = []

        if use_rag:

            rag_results = search_ncert(
                query=current_message,
                class_level=active_class,
                subject=subject,
                chapter=chapter
            )

        ncert_context = (
            build_ncert_context(
                rag_results
            )
        )

        system_prompt = (
            build_system_prompt(
                class_level=active_class,
                subject=subject,
                chapter=chapter,
                ncert_context=ncert_context
            )
        )

        quiz_request = is_quiz_request(
            current_message
        )

        if quiz_request:
            quiz_prompt = build_quiz_prompt(
            class_level=active_class,
            subject=subject,
            chapter=chapter,
            ncert_context=ncert_context,
            user_message=current_message
            )

            messages = [{
                "role": "system",
                "content": quiz_prompt
            }]

        else:

            messages = [{
            "role": "system",
            "content": system_prompt
            }]

        history = request.history [-MAX_HISTORY_MESSAGES:]

        for item in history:

            if item.role not in {
                "user",
                "assistant"
            }:
                continue

            content = (
                item.content.strip()
            )

            if not content:
                continue

            messages.append(
                {
                    "role": item.role,
                    "content": content
                }
            )

        if current_message:

            messages.append(
                {
                    "role": "user",
                    "content":
                        current_message
                }
            )

        if request.attachments:
            for attachment in request.attachments:
                attachment_text = extract_attachment_text(attachment)
                content = f"Attached file: {attachment.name}"
                if attachment_text:
                    content += (
                        "\nExtracted file content:\n"
                        f"{attachment_text}"
                    )
                messages.append({
                    "role": "user",
                    "content": content,
                })

        generation_start = (
            time.perf_counter()
        )

        if quiz_request:
            requested_count = extract_requested_question_count(current_message)
            max_tokens = get_quiz_max_tokens(requested_count)
        elif use_rag:
            max_tokens = 1500
        else:
            max_tokens = 1000

        response = call_gemini(
            messages,
            max_tokens=max_tokens,
            temperature=0.2,
            response_mime_type=(
                "application/json" if quiz_request else None
            ),
        )

        gemini_time = (
            time.perf_counter()
            - generation_start
        )

        reply = response.text or ""

        quiz_data = None

        if quiz_request:
            quiz_data = parse_quiz_response(reply)

            got = (
                len(quiz_data["quiz"]["questions"])
                if quiz_data
                else 0
            )

            finish_reason = (
                response.candidates[0].finish_reason
                if response.candidates
                else None
            )

            print(
                f"Quiz first pass: {got}/{requested_count} "
                f"| finish={finish_reason}"
            )

            # The model sometimes stops early (e.g. 14 of 20) or
            # returns truncated JSON — retry once before falling back.
            if quiz_data is None or got < requested_count:
                print(
                    "Quiz incomplete — retrying once with a "
                    "stricter count reminder..."
                )

                retry_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "REMINDER: Return EXACTLY "
                            f"{requested_count} questions in ONE valid "
                            "JSON object with type quiz. "
                            "Do not stop early and do not truncate. "
                            "Keep each hint and explanation under "
                            "15 words. Output only the JSON."
                        ),
                    }
                ]

                try:
                    retry_response = call_gemini(
                        retry_messages,
                        max_tokens=max_tokens,
                        temperature=0.3,
                        response_mime_type="application/json",
                    )
                    retry_reply = retry_response.text or ""
                    retry_quiz = parse_quiz_response(retry_reply)

                    if retry_quiz:
                        retry_count = len(
                            retry_quiz["quiz"]["questions"]
                        )
                        if quiz_data is None or retry_count > got:
                            quiz_data = retry_quiz
                            reply = retry_reply
                            got = retry_count

                    print(
                        f"Quiz retry: {got}/{requested_count}"
                    )

                except Exception as retry_error:
                    print(
                        "Quiz retry failed:",
                        repr(retry_error),
                    )

            if quiz_data:
                # Guarantee the requested number of questions.
                quiz_data = top_up_quiz_questions(
                    quiz_data,
                    requested_count,
                    active_class,
                    subject,
                    chapter,
                    current_message,
                )
            else:
                quiz_data = generate_fallback_mcq_quiz(
                    active_class,
                    subject,
                    chapter,
                    current_message,
                )

        if request.chat_id:

            old_history = list(
                request.history
            )

            if current_message:

                old_history.append(
                    {
                        "role": "user",
                        "content":
                            current_message
                    }
                )

            if quiz_request and quiz_data:
                # Store a friendly summary instead of raw quiz JSON.
                stored_reply = (
                    "Interactive quiz: "
                    + str(
                        quiz_data["quiz"].get("title", "Quiz")
                    )
                )
            else:
                stored_reply = reply

            old_history.append(
                {
                    "role": "assistant",
                    "content": stored_reply
                }
            )

            saved_chats[request.chat_id] = {
                "id":
                    request.chat_id,
                "title": (
                    request.chat_title
                    or "New chat"
                ),
                "updated_at":
                    datetime.now().isoformat(),
                "messages":
                    old_history,
                "class_level":
                    active_class
            }

        if quiz_request and quiz_data:

            total_time = (
            time.perf_counter()
            - total_start
            )

            actual_count = len(
                quiz_data["quiz"]["questions"]
            )

            print(
            f"Quiz generated: {actual_count}/{requested_count} | "
            f"Gemini: {gemini_time:.2f}s | "
            f"Total: {total_time:.2f}s"
            )

            return {
                "type": "quiz",
                "quiz": quiz_data["quiz"],
                "message": build_quiz_intro_message(
                    actual_count,
                    requested_count,
                ),
                "class_level": active_class,
                "subject": subject,
                "chapter": chapter,
                "rag": {
                    "used": len(rag_results) > 0,
                    "sources": len(rag_results)
                    }
                }

        if (
            quiz_request
            and not quiz_data
            and _looks_like_quiz_json(reply)
        ):
            # Never show raw JSON text to the student.
            reply = (
                "Oops! 😅 I couldn't build a valid quiz that time. "
                "Could you please ask me again? I'll regenerate "
                "it for you! 🍀"
            )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"Gemini: {gemini_time:.2f}s | "
            f"Total: {total_time:.2f}s"
        )

        return {
            "reply": reply,
            "class_level":
                active_class,
            "subject":
                subject,
            "chapter":
                chapter,
            "rag": {
                "used":
                    len(rag_results) > 0,
                "sources":
                    len(rag_results)
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "\n========== CHAT ERROR =========="
        )

        print(
            repr(e)
        )

        print(
            "================================\n"
        )

        raise HTTPException(
            status_code=500,
            detail="AI request failed."
        )


@app.get("/api/chats")
def get_chats():

    return list(
        saved_chats.values()
    )


@app.get("/api/chats/{chat_id}")
def get_chat(
    chat_id: str
):

    if chat_id not in saved_chats:

        raise HTTPException(
            status_code=404,
            detail="Chat not found"
        )

    return saved_chats[chat_id]


@app.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str
):

    if chat_id not in saved_chats:

        raise HTTPException(
            status_code=404,
            detail="Chat not found"
        )

    del saved_chats[chat_id]

    return {
        "success": True
    }