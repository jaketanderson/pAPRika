"""
Functions for generate GAFF files and parameters.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import parmed as pmd
import simtk.unit as simtk_unit
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk.openmm.app import AmberPrmtopFile
from simtk.openmm.app import element as E
from simtk.openmm.app.internal.customgbforces import _get_bonded_atom_list

from paprika import unit
from paprika.build.system import TLeap
from paprika.utils.utils import is_number

logger = logging.getLogger(__name__)
_PI_ = np.pi


def generate_gaff(
    mol2_file: str,
    residue_name: str,
    output_name: Optional[str] = None,
    need_gaff_atom_types: Optional[bool] = True,
    generate_frcmod: Optional[bool] = True,
    directory_path: Optional[str] = "benchmarks",
    gaff_version: Optional[str] = "gaff2",
):
    """
    Module to generate GAFF files given a mol2 file.

    Parameters
    ----------
    mol2_file
        The name of the mol2 structure file.
    residue_name
        The residue name of the molecule.
    output_name
        The name for the output file.
    need_gaff_atom_types
        Whether to generate GAFF atoms or not. Currently, this is the only choice.
    gaff_version
        The GAFF version to use ("gaff1", "gaff2")
    generate_frcmod
        Option to generate a GAFF frcmod file.
    directory_path
        The working directory where the files will be stored.
    """

    if output_name is None:
        output_name = mol2_file.stem

    if need_gaff_atom_types:
        _generate_gaff_atom_types(
            mol2_file=mol2_file,
            residue_name=residue_name,
            output_name=output_name,
            gaff_version=gaff_version,
            directory_path=directory_path,
        )
        logging.debug(
            "Checking to see if we have a multi-residue MOL2 file that should be converted "
            "to single-residue..."
        )
        structure = pmd.load_file(
            os.path.join(directory_path, f"{output_name}.{gaff_version}.mol2"),
            structure=True,
        )
        if len(structure.residues) > 1:
            structure[":1"].save("tmp.mol2")
            if os.path.exists("tmp.mol2"):
                os.rename(
                    "tmp.mol2",
                    os.path.join(directory_path, f"{output_name}.{gaff_version}.mol2"),
                )
                logging.debug("Saved single-residue MOL2 file for `tleap`.")
            else:
                raise RuntimeError(
                    "Unable to convert multi-residue MOL2 file to single-residue for `tleap`."
                )

        if generate_frcmod:
            _generate_frcmod(
                mol2_file=f"{output_name}.{gaff_version}.mol2",
                gaff_version=gaff_version,
                output_name=output_name,
                directory_path=directory_path,
            )

    else:
        raise NotImplementedError()


def _generate_gaff_atom_types(
    mol2_file: str,
    residue_name: str,
    output_name: str,
    gaff_version: Optional[str] = "gaff2",
    directory_path: Optional[str] = "benchmarks",
):
    """Generate a mol2 file with GAFF atom types."""

    if gaff_version.lower() not in ["gaff", "gaff2"]:
        raise KeyError(
            f"Parameter set {gaff_version} not supported. Only [gaff, gaff2] are allowed."
        )

    p = subprocess.Popen(
        [
            "antechamber",
            "-i",
            mol2_file,
            "-fi",
            "mol2",
            "-o",
            f"{output_name}.{gaff_version}.mol2",
            "-fo",
            "mol2",
            "-rn",
            f"{residue_name.upper()}",
            "-at",
            f"{gaff_version}",
            "-an",
            "no",
            "-dr",
            "no",
            "-pf",
            "yes",
        ],
        cwd=directory_path,
    )
    p.communicate()
    print(p)

    remove_files = [
        "ANTECHAMBER_AC.AC",
        "ANTECHAMBER_AC.AC0",
        "ANTECHAMBER_BOND_TYPE.AC",
        "ANTECHAMBER_BOND_TYPE.AC0",
        "ATOMTYPE.INF",
    ]
    files = [os.path.join(directory_path, file) for file in remove_files]
    for file in files:
        if os.path.isfile(file):
            logger.debug(f"Removing temporary file: {file}")
            file.unlink()

    if not os.path.exists(f"{output_name}.{gaff_version}.mol2"):
        # Try with the newer (AmberTools 19) version of `antechamber` which doesn't have the `-dr` flag
        p = subprocess.Popen(
            [
                "antechamber",
                "-i",
                mol2_file,
                "-fi",
                "mol2",
                "-o",
                f"{output_name}.{gaff_version}.mol2",
                "-fo",
                "mol2",
                "-rn",
                f"{residue_name.upper()}",
                "-at",
                f"{gaff_version}",
                "-an",
                "no",
                "-pf",
                "yes",
            ],
            cwd=directory_path,
        )
        p.communicate()

        remove_files = [
            "ANTECHAMBER_AC.AC",
            "ANTECHAMBER_AC.AC0",
            "ANTECHAMBER_BOND_TYPE.AC",
            "ANTECHAMBER_BOND_TYPE.AC0",
            "ATOMTYPE.INF",
        ]
        files = [os.path.join(directory_path, file) for file in remove_files]
        for file in files:
            if os.path.isfile(file):
                logger.debug(f"Removing temporary file: {file}")
                file.unlink()


def _generate_frcmod(
    mol2_file: str,
    gaff_version: str,
    output_name: str,
    directory_path: Optional[str] = "benchmarks",
):
    """Generate an AMBER .frcmod file given a mol2 file."""

    if gaff_version.lower() not in ["gaff", "gaff2"]:
        raise KeyError(
            f"Parameter set {gaff_version} not supported. Only [gaff, gaff2] are allowed."
        )

    subprocess.Popen(
        [
            "parmchk2",
            "-i",
            str(mol2_file),
            "-f",
            "mol2",
            "-o",
            f"{output_name}.{gaff_version}.frcmod",
            "-s",
            f"{gaff_version}",
        ],
        cwd=directory_path,
    )


class GAFFForceField:
    @property
    def smiles_list(self):
        """list: A list containing the smiles string of the system substances."""
        return self._smiles_list

    @smiles_list.setter
    def smiles_list(self, value: list):
        self._smiles_list = value

    @property
    def library_charges(self):
        """LibraryChargeHandler:"""
        force_field = ForceField(self._library_charges)
        return force_field.get_parameter_handler("LibraryCharges")

    @library_charges.setter
    def library_charges(self, value: str):
        self._library_charges = value

    @property
    def gaff_version(self):
        """str: The version of GAFF to use (`gaff` or `gaff2`)."""
        return self._gaff_version

    @gaff_version.setter
    def gaff_version(self, value):
        self._gaff_version = value

    @property
    def cutoff(self):
        """pint.unit.Quantity: The non-bonded interaction cutoff."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: unit.Quantity):
        self._cutoff = value

    @property
    def igb(self):
        """int: The Amber Generalized Born Implicit Solvent (GBIS) model."""
        return self._igb

    @igb.setter
    def igb(self, value):
        self._igb = value

    @property
    def sa_model(self):
        """str: The surface area (SA) model for running GB/SA simulation."""
        return self._sa_model

    @sa_model.setter
    def sa_model(self, value):
        self._sa_model = value

    @property
    def topology(self):
        """ParmEd.structure: The topology of the system as a ParmEd object."""
        return self._topology

    @property
    def frcmod_parameters(self):
        """dict: The `frcmod` parameters stored in a dictionary."""
        return self._frcmod_parameters

    @frcmod_parameters.setter
    def frcmod_parameters(self, value):
        self._frcmod_parameters = value

    def __init__(
        self,
        smiles_list=None,
        library_charges=None,
        gaff_version="gaff",
        cutoff=9.0 * unit.angstrom,
        frcmod_parameters=None,
        igb=None,
        sa_model="ACE",
    ):
        if gaff_version.lower() not in ["gaff", "gaff2"]:
            raise KeyError(
                f"Parameter set {gaff_version} not supported. Only [gaff, gaff2] are allowed."
            )

        self._smiles_list = smiles_list
        self._library_charges = library_charges
        self._gaff_version = gaff_version
        self._cutoff = cutoff
        self._igb = igb
        self._sa_model = sa_model

        self._topology = None
        self._frcmod_parameters = (
            self.frcmod_file_to_dict(frcmod_parameters)
            if isinstance(frcmod_parameters, str)
            else frcmod_parameters
        )

        if smiles_list is not None:
            self.initialize()

    def initialize(self):
        # Extract GAFF parameters
        working_directory = tempfile.mkdtemp()
        molecule_list = []

        for i, smiles in enumerate(self._smiles_list):
            # Generate mol2 file
            molecule = Molecule.from_smiles(smiles)
            molecule.partial_charges = (
                np.zeros(molecule.n_atoms) * simtk_unit.elementary_charge
            )
            molecule.to_file(
                os.path.join(working_directory, f"MOL{i}.mol2"),
                file_format="MOL2",
            )

            generate_gaff(
                mol2_file=f"MOL{i}.mol2",
                residue_name=f"MOL{i}",
                output_name=f"MOL{i}",
                need_gaff_atom_types=True,
                generate_frcmod=False,
                directory_path=working_directory,
                gaff_version=self.gaff_version,
            )

            # Generate prmtop file
            system = TLeap()
            system.output_path = working_directory
            system.output_prefix = f"MOL{i}.{self.gaff_version}"
            system.pbc_type = None
            system.neutralize = False
            system.template_lines = [
                f"source leaprc.{self.gaff_version}",
                f"MOL{i} = loadmol2 MOL{i}.{self.gaff_version}.mol2",
                f"saveamberparm MOL{i} {system.output_prefix}.prmtop {system.output_prefix}.rst7",
                "quit",
            ]
            system.build(clean_files=False, ignore_warnings=True)

            molecule_list.append(
                os.path.join(working_directory, f"{system.output_prefix}.prmtop")
            )

        # Generate OpenMM topology
        topology = pmd.load_file(molecule_list[0], structure=True)
        for molecule in molecule_list[1:]:
            topology += pmd.load_file(molecule, structure=True)
        topology.save(os.path.join(working_directory, "full.prmtop"), overwrite=True)
        self._topology = AmberPrmtopFile(os.path.join(working_directory, "full.prmtop"))

        # Generate full frcmod file
        pmd.tools.writeFrcmod(
            topology,
            os.path.join(working_directory, "complex.frcmod"),
        ).execute()
        self._frcmod_parameters = GAFFForceField.frcmod_file_to_dict(
            os.path.join(working_directory, "complex.frcmod")
        )

        # Delete temp folder
        shutil.rmtree(working_directory)

        if self.igb:
            all_bonds = _get_bonded_atom_list(self._topology.topology)

            # Apply `mbondi` radii (igb=1)
            if self.igb == 1:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element in (E.carbon, E.nitrogen):
                            radii = 1.3
                            mask = "H-C" if bondeds[0].element is E.carbon else "H-N"
                        elif bondeds[0].element in (E.oxygen, E.sulfur):
                            radii = 0.8
                            mask = "H-O" if bondeds[0].element is E.oxygen else "H-S"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii / 10,
                                    "cosmetic": None,
                                }
                            }
                        )

            # Apply `mbondi2` radii (igb=2,5)
            elif self.igb in [2, 5]:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element is E.nitrogen:
                            radii = 1.3
                            mask = "H-N"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depeends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii / 10,
                                    "cosmetic": None,
                                }
                            }
                        )

    def get_parameter_value(self, tag, atom_mask, *attributes):
        """Returns an FF parameter(s) as a dictionary. Multiple parameters
        can be returned for a specific tag.

        Parameters
        ----------
        tag: str
           FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attributes: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).

        Returns
        -------
        parameter: dict
            A dictionary with the FF parameter.
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is not listed under `{tag}`.")

        parameter = {tag: {atom_mask: {}}}
        for attribute in attributes:
            if attribute in self._frcmod_parameters[tag][atom_mask]:
                parameter[tag][atom_mask].update(
                    {attribute: self._frcmod_parameters[tag][atom_mask][attribute]}
                )
            else:
                raise KeyError(
                    f"`{attribute}` is not an attribute of `{tag}-{atom_mask}`."
                )

        return parameter

    def set_parameter_value(self, tag, atom_mask, attribute, value):
        """Set the value for a FF parameter.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attribute: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).
        value: float
            The value for the FF parameter.
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")

        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tag}`.")

        if attribute not in self._frcmod_parameters[tag][atom_mask]:
            raise KeyError(
                f"The attribute `{attribute}` is not an attribute of `{tag}-{atom_mask}`."
            )

        self._frcmod_parameters[tag][atom_mask][attribute] = value

    def tag_parameter_to_optimize(self, tag, atom_mask, *attributes):
        """Tag a FF parameter(s) for use in a ForceBalance run. When writing
        to file, the tagged FF parameter(s) will have a comment "# PRM ..."
        at then end of the line.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attributes: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tag}`.")

        cosmetic = "# PRM"
        for attribute in attributes:
            if attribute in self._frcmod_parameters[tag][atom_mask]:

                if tag == "BOND":
                    if attribute == "k":
                        cosmetic += " 1"
                    elif attribute == "length":
                        cosmetic += " 2"

                elif tag == "ANGLE":
                    if attribute == "k":
                        cosmetic += " 1"
                    elif attribute == "angle":
                        cosmetic += " 2"

                elif tag == "DIHEDRAL":
                    if attribute == "scaling":
                        cosmetic += " 1"
                    elif attribute == "barrier":
                        cosmetic += " 2"
                    elif attribute == "phase":
                        cosmetic += " 3"
                    elif attribute == "periodicity":
                        cosmetic += " 4"

                elif tag == "IMPROPER":
                    if attribute == "barrier":
                        cosmetic += " 1"
                    elif attribute == "phase":
                        cosmetic += " 2"
                    elif attribute == "periodicity":
                        cosmetic += " 3"

                elif tag == "VDW":
                    if attribute == "rmin_half":
                        cosmetic += " 1"
                    elif attribute == "epsilon":
                        cosmetic += " 2"

                elif tag == "GBSA":
                    if attribute == "radius":
                        cosmetic += " 1"

            else:
                raise KeyError(
                    f"`{attribute}` is not an attribute of `{tag}-{atom_mask}`."
                )

        self._frcmod_parameters[tag][atom_mask]["cosmetic"] = cosmetic

    @staticmethod
    def _parameter_to_string(tag, atom_mask, parameters):
        """Convert a parameter to a string in AMBER frcmod file format.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        parameters: dict
            A dictionary containing the FF attribute and parameters.

        Return
        ------
        parameter_line: str
            A string with the FF parameter in AMBER frcmod format (https://ambermd.org/FileFormats.php#frcmod).
        """
        parameter_line = None

        if tag == "MASS":
            parameter_line = f"{atom_mask:2s}"
            parameter_line += f"{parameters['mass']:10.3f}"

        if tag == "BOND":
            parameter_line = f"{atom_mask:5s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['length']:10.3f}"

        if tag == "ANGLE":
            parameter_line = f"{atom_mask:8s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['theta']:10.3f}"

        if tag == "DIHEDRAL":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['scaling']:4d}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"
            if parameters["SCEE"]:
                parameter_line += f"  SCEE={parameters['SCEE']:.1f}"
            if parameters["SCNB"]:
                parameter_line += f"  SCNB={parameters['SCNB']:.1f}"

        if tag == "IMPROPER":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"

        if tag == "VDW":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['rmin_half']:15.8f}"
            parameter_line += f"{parameters['epsilon']:15.8f}"

        if tag == "GBSA":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['radius']:15.8f}"

        if parameters["cosmetic"]:
            parameter_line += f"   {parameters['cosmetic']}"

        assert parameter_line is not None

        parameter_line += "\n"

        return parameter_line

    def to_file(self, file_path, write_header=False, skip_gbsa=True):
        """Write the FF parameters to an AMBER frcmod file.

        Parameters
        ----------
        file_path: str
            The name of the frcmod file.
        write_header: bool
            Whether to print header information (used in ForceBalance runs).
        skip_gbsa: bool
            Whether to skip printing FF parameters for GBSA (not read in TLeap but used in ForceBalance runs).
        """

        with open(file_path, "w") as f:

            for tag in self._frcmod_parameters.keys():
                if tag == "HEADER" and write_header:
                    f.writelines(
                        "#evaluator_io: "
                        f"gaff_version={self._gaff_version} "
                        f"cutoff={self._cutoff.magnitude} "
                        f"igb={self._igb} "
                        f"sa_model={self._sa_model} \n"
                    )
                    continue
                elif tag == "HEADER" and not write_header:
                    f.writelines("Remark line goes here\n")
                    continue

                if tag == "GBSA" and skip_gbsa:
                    continue

                if tag == "DIHEDRAL":
                    f.writelines("DIHE\n")
                elif tag == "VDW":
                    f.writelines("NONBON\n")
                else:
                    f.writelines(f"{tag}\n")

                for atom_mask in self._frcmod_parameters[tag]:
                    f.writelines(
                        self._parameter_to_string(
                            tag,
                            atom_mask,
                            self._frcmod_parameters[tag][atom_mask],
                        )
                    )
                f.writelines("\n")

    @classmethod
    def from_file(cls, file_path: str):
        """Create an instance of this class by reading in a frcmod file."""
        frcmod_pdict = cls.frcmod_file_to_dict(file_path)
        gaff_version = "gaff"
        cutoff = 9.0 * unit.angstrom
        igb = None
        sa_model = None

        if frcmod_pdict["HEADER"]:
            gaff_version = frcmod_pdict["HEADER"]["leap_source"]
            cutoff = frcmod_pdict["HEADER"]["cutoff"] * unit.angstrom
            igb = int(frcmod_pdict["HEADER"]["igb"])
            sa_model = (
                None
                if frcmod_pdict["HEADER"]["sa_model"] == "None"
                else frcmod_pdict["HEADER"]["sa_model"]
            )

        new_instance = cls(
            smiles_list=None,
            gaff_version=gaff_version,
            cutoff=cutoff,
            igb=igb,
            sa_model=sa_model,
        )
        new_instance.frcmod_parameters = frcmod_pdict

        return new_instance

    @staticmethod
    def frcmod_file_to_dict(file_path: str) -> dict:
        """Read in a frcmod file and stores the information in a dictionary.

        .. note ::
            Parameters with polarizabilities are not supported yet and will be ignored.

        Parameters
        ----------
        file_path: str
            The fcmod file to process.

        Returns
        -------
        frcmod_dict: dict
            A dictionary containing the parameters from the frcmod file.
        """

        frcmod_dict = {
            "HEADER": {},
            "MASS": {},
            "BOND": {},
            "ANGLE": {},
            "DIHEDRAL": {},
            "IMPROPER": {},
            "VDW": {},
            "GBSA": {},
        }

        with open(file_path, "r") as f:

            for i, line in enumerate(f.readlines()):

                if i == 0 and line.startswith("#evaluator_io:"):
                    header = line.split()
                    frcmod_dict["HEADER"] = {
                        "leap_source": header[1].split("=")[-1],
                        "cutoff": float(header[2].split("=")[-1]),
                        "igb": int(header[3].split("=")[-1]),
                        "sa_model": header[4].split("=")[-1],
                    }
                    continue

                if (
                    (i == 0 and not line.startswith("#evaluator:"))
                    or line.strip() == 0
                    or line.startswith("\n")
                ):
                    continue

                if re.match("MASS", line.strip().upper()):
                    keyword = "MASS"
                    continue
                elif re.match("BOND|BONDS", line.strip().upper()):
                    keyword = "BOND"
                    continue
                elif re.match("ANGLE|ANGLES", line.strip().upper()):
                    keyword = "ANGLE"
                    continue
                elif re.match("DIHE|DIHEDRAL|DIHEDRALS", line.strip().upper()):
                    keyword = "DIHEDRAL"
                    continue
                elif re.match("IMPROPER", line.strip().upper()):
                    keyword = "IMPROPER"
                    continue
                elif re.match("NONBON|NONB|NONBONDED", line.strip().upper()):
                    keyword = "VDW"
                    continue
                elif re.match("RADII|GBSA|GBRADII", line.strip().upper()):
                    keyword = "GBSA"
                    continue

                # Read parameter
                cosmetic = None
                parameter = line.split()
                if "#" in line:
                    parameter = line[: line.index("#")].split()
                    cosmetic = line[line.index("#") :]

                atom_columns = []
                for j in range(len(parameter)):
                    # Convert to float
                    if is_number(parameter[j]) and not parameter[j].isdigit():
                        parameter[j] = float(parameter[j])

                    # Convert to int
                    elif is_number(parameter[j]) and parameter[j].isdigit():
                        parameter[j] = int(parameter[j])

                    # Get list element that are strings
                    elif "SC" not in parameter[j]:
                        atom_columns.append(j)

                # Get proper formatting for atom masks
                mask = parameter[0]
                if len(atom_columns) > 1:
                    atom_mask = "".join(parameter[: len(atom_columns)])
                    for k, col in enumerate(atom_columns):
                        parameter.remove(parameter[col - k])
                    mask = "-".join(f"{atom:2s}" for atom in atom_mask.split("-"))
                else:
                    parameter.pop(0)

                # Build parameter dictionary
                if keyword == "MASS":
                    param_dict = {"mass": parameter[0], "cosmetic": cosmetic}

                elif keyword == "BOND":
                    param_dict = {
                        "k": parameter[0],
                        "length": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "ANGLE":
                    param_dict = {
                        "k": parameter[0],
                        "theta": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "DIHEDRAL":
                    param_dict = {
                        "scaling": parameter[0],
                        "barrier": parameter[1],
                        "phase": parameter[2],
                        "periodicity": parameter[3],
                        "SCEE": None,
                        "SCNB": None,
                        "cosmetic": cosmetic,
                    }
                    if len(parameter) > 4:
                        if "SCEE" in parameter[4]:
                            param_dict["SCEE"] = float(parameter[4].split("=")[1])
                        if "SCNB" in parameter[4]:
                            param_dict["SCNB"] = float(parameter[4].split("=")[1])

                    if len(parameter) > 5:
                        if "SCEE" in parameter[5]:
                            param_dict["SCEE"] = float(parameter[5].split("=")[1])
                        if "SCNB" in parameter[5]:
                            param_dict["SCNB"] = float(parameter[5].split("=")[1])

                elif keyword == "IMPROPER":
                    param_dict = {
                        "barrier": parameter[0],
                        "phase": parameter[1],
                        "periodicity": parameter[2],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "VDW":
                    param_dict = {
                        "rmin_half": parameter[0],
                        "epsilon": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "GBSA":
                    param_dict = {"radius": parameter[0], "cosmetic": cosmetic}

                # Update dictionary
                frcmod_dict[keyword].update({mask: param_dict})

        return frcmod_dict

    def to_json(self, file_path: str):
        """Save current FF parameters to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self._frcmod_parameters, f)

    @classmethod
    def from_json(cls, file_path: str):
        """Create an instance of this class by reading in a JSON file."""
        with open(file_path, "r") as f:
            frcmod_pdict = json.load(f)

        gaff_version = "gaff"
        cutoff = 9.0 * unit.angstrom
        igb = None
        sa_model = None

        if frcmod_pdict["HEADER"]:
            gaff_version = frcmod_pdict["HEADER"]["leap_source"]
            cutoff = frcmod_pdict["HEADER"]["cutoff"] * unit.angstrom
            igb = int(frcmod_pdict["HEADER"]["igb"])
            sa_model = (
                None
                if frcmod_pdict["HEADER"]["sa_model"] == "None"
                else frcmod_pdict["HEADER"]["sa_model"]
            )

        new_instance = cls(
            smiles_list=None,
            gaff_version=gaff_version,
            cutoff=cutoff,
            igb=igb,
            sa_model=sa_model,
        )
        new_instance.frcmod_parameters = frcmod_pdict

        return new_instance
