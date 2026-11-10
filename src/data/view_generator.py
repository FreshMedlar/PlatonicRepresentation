"""
Synthetic Good-View Generator for Text JEPA.
Generates invariant world-state propositions with independent modality nuisances
(syntax, vocabulary, tone, style) and multi-step reasoning trajectories.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PropositionState:
    state_id: str
    concept: str
    views: Dict[str, str]  # style -> text representation


@dataclass
class ReasoningTrajectory:
    trajectory_id: str
    topic: str
    steps: List[PropositionState]  # Sequential semantic chunks (x^(1), ..., x^(T))


class GoodViewGenerator:
    """
    Generates synthetic semantic propositions rendered into independent stylistic views,
    as well as sequential multi-step reasoning traces.
    """

    STYLES = ["standard", "formal", "casual", "concise"]

    # Base parametric templates for semantic concepts
    TEMPLATES = [
        # Physics / Thermodynamics
        {
            "concept": "water_freezing",
            "standard": "Water freezes into solid ice at zero degrees Celsius.",
            "formal": "The phase transition of liquid water to solid ice occurs at approximately zero degrees Celsius under standard atmospheric conditions.",
            "casual": "Whenever water hits zero degrees Celsius it turns into solid ice.",
            "concise": "Water freezes at 0 C.",
        },
        {
            "concept": "water_boiling",
            "standard": "Water reaches its boiling point at one hundred degrees Celsius.",
            "formal": "Thermal vaporization of water occurs at one hundred degrees Celsius under standard barometric pressure.",
            "casual": "Water boils and creates steam once it gets to one hundred degrees C.",
            "concise": "Water boiling point is 100 C.",
        },
        {
            "concept": "gravitational_attraction",
            "standard": "Massive objects exert a gravitational force that attracts other bodies.",
            "formal": "Gravitational attraction between physical bodies is directly proportional to the product of their masses.",
            "casual": "Heavy things pull other things towards them because of gravity.",
            "concise": "Mass generates gravitational pull.",
        },
        {
            "concept": "speed_of_light",
            "standard": "Light travels through a vacuum at three hundred thousand kilometers per second.",
            "formal": "The velocity of electromagnetic radiation in a vacuum is invariant at approximately 300,000 kilometers per second.",
            "casual": "Light zips through empty space at about three hundred thousand kilometers every second.",
            "concise": "Speed of light is 300,000 km/s.",
        },
        {
            "concept": "photosynthesis",
            "standard": "Plants convert sunlight and carbon dioxide into oxygen and chemical energy.",
            "formal": "Phototrophic organisms synthesize carbohydrates from carbon dioxide and water utilizing radiant solar energy.",
            "casual": "Green plants soak up sunshine and carbon dioxide to make food and oxygen.",
            "concise": "Photosynthesis turns light into oxygen and energy.",
        },
        {
            "concept": "earth_orbit",
            "standard": "The Earth completes one full orbit around the Sun in approximately three hundred sixty-five days.",
            "formal": "The heliocentric orbital revolution of Earth requires roughly three hundred sixty-five solar days.",
            "casual": "It takes our planet about a year to go all the way around the Sun.",
            "concise": "Earth orbits Sun in 365 days.",
        },
        {
            "concept": "cellular_respiration",
            "standard": "Cells break down glucose in the presence of oxygen to produce cellular ATP.",
            "formal": "Aerobic cellular respiration catabolizes glucose molecules to synthesize adenosine triphosphate.",
            "casual": "Cells consume glucose and oxygen to generate the energy they need to live.",
            "concise": "Cells make ATP from glucose and oxygen.",
        },
        {
            "concept": "sound_propagation",
            "standard": "Sound requires a physical medium such as air or water to propagate vibrations.",
            "formal": "Acoustic mechanical waves necessitate a material medium for longitudinal wave transmission.",
            "casual": "Sound waves cannot travel through a vacuum because they need air or liquid to vibrate.",
            "concise": "Sound requires a medium to travel.",
        },
        {
            "concept": "dna_structure",
            "standard": "Deoxyribonucleic acid is composed of two nucleotide strands arranged in a double helix.",
            "formal": "The primary macromolecular architecture of DNA is an antiparallel double helical polymer.",
            "casual": "DNA looks like a twisted ladder made of genetic building blocks.",
            "concise": "DNA has a double helix structure.",
        },
        {
            "concept": "friction_heat",
            "standard": "Friction between rubbing surfaces converts mechanical kinetic energy into thermal heat.",
            "formal": "Mechanical resistance across contacting interfaces dissipates kinetic energy as thermal energy.",
            "casual": "Rubbing two surfaces together produces friction and makes them warm up.",
            "concise": "Friction converts kinetic motion into heat.",
        },
    ]

    # Trajectories of multi-step logical chains
    TRAJECTORY_TEMPLATES = [
        {
            "topic": "evaporative_cooling",
            "steps": [
                {
                    "concept": "liquid_heated",
                    "standard": "Thermal energy is supplied to the liquid container.",
                    "formal": "Exogenous thermal energy is introduced into the liquid reservoir.",
                    "casual": "Heat gets added to the liquid inside the pot.",
                    "concise": "Heat enters liquid.",
                },
                {
                    "concept": "molecules_gain_speed",
                    "standard": "Surface molecules gain kinetic speed and overcome intermolecular bonds.",
                    "formal": "High-energy particles at the interface overcome intermolecular cohesion forces.",
                    "casual": "Molecules near the surface move faster and break free into air.",
                    "concise": "Molecules escape surface.",
                },
                {
                    "concept": "latent_heat_removal",
                    "standard": "Escaping vapor carries away thermal energy from the remaining liquid.",
                    "formal": "Vaporization removes latent thermal energy from the remaining liquid substrate.",
                    "casual": "The escaping steam takes warmth with it as it leaves.",
                    "concise": "Vapor carries heat away.",
                },
                {
                    "concept": "cooling_result",
                    "standard": "Consequently the temperature of the remaining liquid decreases.",
                    "formal": "Therefore the average kinetic temperature of the liquid decreases steadily.",
                    "casual": "So the liquid left behind ends up getting cooler.",
                    "concise": "Remaining liquid cools.",
                },
            ],
        },
        {
            "topic": "rain_formation",
            "steps": [
                {
                    "concept": "solar_evaporation",
                    "standard": "Solar radiation warms oceanic water causing evaporation into the atmosphere.",
                    "formal": "Radiant solar flux heats surface water causing vapor ascent into the troposphere.",
                    "casual": "The Sun warms the ocean so water evaporates into the sky.",
                    "concise": "Sun evaporates ocean water.",
                },
                {
                    "concept": "vapor_condensation",
                    "standard": "Warm water vapor rises into cooler air and condenses into cloud droplets.",
                    "formal": "Ascending humid air undergoes adiabatic cooling, precipitating microscopic droplets.",
                    "casual": "Rising humid vapor cools off and condenses into fluffy clouds.",
                    "concise": "Vapor condenses into clouds.",
                },
                {
                    "concept": "droplet_coalescence",
                    "standard": "Small droplets collide and coalesce into heavier raindrops within the cloud.",
                    "formal": "Microscopic droplets coalesce via turbulence into macroscopic precipitation drops.",
                    "casual": "The tiny water drops bump together until they become heavy drops.",
                    "concise": "Droplets merge into drops.",
                },
                {
                    "concept": "precipitation_fall",
                    "standard": "Gravity pulls the heavy raindrops downward to the surface as rainfall.",
                    "formal": "Gravitational acceleration overcomes atmospheric updrafts causing surface precipitation.",
                    "casual": "Once they get too heavy, gravity pulls them down as rain.",
                    "concise": "Rain falls by gravity.",
                },
            ],
        },
        {
            "topic": "neural_action_potential",
            "steps": [
                {
                    "concept": "resting_potential",
                    "standard": "The neuronal membrane maintains a negative resting electrical potential.",
                    "formal": "The resting neuronal plasma membrane exhibits polarized negative membrane potential.",
                    "casual": "Before firing, the nerve cell keeps a negative electrical charge inside.",
                    "concise": "Neuron holds resting charge.",
                },
                {
                    "concept": "threshold_depolarization",
                    "standard": "Excitatory synaptic input depolarizes the membrane toward the threshold voltage.",
                    "formal": "Excitatory neurotransmission induces graded depolarization exceeding threshold potential.",
                    "casual": "An incoming signal pushes the voltage up toward the firing threshold.",
                    "concise": "Signal triggers threshold voltage.",
                },
                {
                    "concept": "sodium_influx",
                    "standard": "Voltage-gated sodium ion channels open rapidly allowing sodium ions to rush inside.",
                    "formal": "Voltage-sensitive sodium conductance channels open, driving massive inward cationic flux.",
                    "casual": "Sodium channels swing wide open and positive ions flood into the cell.",
                    "concise": "Sodium ions rush in.",
                },
                {
                    "concept": "spike_propagation",
                    "standard": "The sudden electrical action potential propagates along the axon to the synapse.",
                    "formal": "An all-or-none action potential propagates electrotonically along the axon terminal.",
                    "casual": "An electrical spike shoots all the way down the nerve fiber.",
                    "concise": "Action potential travels axon.",
                },
            ],
        },
        {
            "topic": "algorithmic_binary_search",
            "steps": [
                {
                    "concept": "sorted_precondition",
                    "standard": "The search procedure requires the input array elements to be monotonically sorted.",
                    "formal": "Binary search establishes a strict precondition of a sorted linear array index.",
                    "casual": "First ensure the list of numbers is already sorted from low to high.",
                    "concise": "Array must be sorted.",
                },
                {
                    "concept": "midpoint_probe",
                    "standard": "The algorithm computes the midpoint element and compares it against the target key.",
                    "formal": "The index median is evaluated against the target value via ordinal comparison.",
                    "casual": "Look at the exact middle item and check if it matches what you want.",
                    "concise": "Compare middle element to target.",
                },
                {
                    "concept": "halve_search_space",
                    "standard": "Depending on the comparison result, half of the remaining array is discarded.",
                    "formal": "Based on comparative ordering, the active search interval is halved immediately.",
                    "casual": "Throw away whichever half cannot possibly contain your target value.",
                    "concise": "Discard non-matching half.",
                },
                {
                    "concept": "logarithmic_convergence",
                    "standard": "The process terminates with logarithmic computational time complexity.",
                    "formal": "The iterative bifurcation converges with provable logarithmic time complexity.",
                    "casual": "Repeating this narrows down the target in logarithmic time.",
                    "concise": "Converges in O(log n) time.",
                },
            ],
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def get_all_texts(self) -> List[str]:
        """Collects all texts across all concepts, styles, and trajectory steps."""
        corpus = []
        for item in self.TEMPLATES:
            for style in self.STYLES:
                if style in item:
                    corpus.append(item[style])

        for traj in self.TRAJECTORY_TEMPLATES:
            for step in traj["steps"]:
                for style in self.STYLES:
                    if style in step:
                        corpus.append(step[style])
        return corpus

    def sample_positive_pair(self) -> Tuple[str, str, str, str, str]:
        """
        Samples a positive Good-View pair:
        (view1_text, view2_text, concept_id, style1, style2)
        Both views share the exact same concept (world state z),
        while styles (nuisance v) are sampled independently.
        """
        # Pool templates from both standalone and trajectory steps
        pool = []
        for item in self.TEMPLATES:
            pool.append((item["concept"], item))
        for traj in self.TRAJECTORY_TEMPLATES:
            for step in traj["steps"]:
                pool.append((step["concept"], step))

        concept, item = self.rng.choice(pool)
        available_styles = [s for s in self.STYLES if s in item]

        if len(available_styles) >= 2:
            s1, s2 = self.rng.sample(available_styles, 2)
        else:
            s1 = s2 = available_styles[0]

        return item[s1], item[s2], concept, s1, s2

    def sample_reasoning_trajectory(self) -> Tuple[List[str], List[str], str]:
        """
        Samples a sequential reasoning trajectory:
        Returns:
            step_texts: List of semantic chunk strings [x^(1), ..., x^(T)]
            step_styles: List of styles used per chunk
            topic: The overall reasoning topic
        """
        traj = self.rng.choice(self.TRAJECTORY_TEMPLATES)
        topic = traj["topic"]
        step_texts = []
        step_styles = []

        for step in traj["steps"]:
            style = self.rng.choice(self.STYLES)
            step_texts.append(step[style])
            step_styles.append(style)

        return step_texts, step_styles, topic
